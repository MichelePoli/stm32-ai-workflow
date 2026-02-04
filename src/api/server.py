
import os
import uvicorn
import logging
import redis
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.assistant.graph import builder, redis_client, checkpointer_redis
from src.assistant.state import MasterState
from src.assistant.configuration import Configuration
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# Configura Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default-session"

# Global placeholders per il grafo (inizializzati nello startup)
graph = None
memory = None

def format_sse(data: str) -> str:
    """Formatta stringa per Server-Sent Events (opzionale se client gestisce chunk raw)"""
    return f"data: {data}\n\n"

from contextlib import asynccontextmanager

# Inizializzazione Graph & Redis Checkpointer (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inizializza il grafo e il checkpointer Redis all'avvio (dentro l'event loop)."""
    global graph, memory
    logger.info("🚀 Inizializzazione Graph & Redis Checkpointer...")
    try:
        # 1. Crea il checkpointer asincrono (ora che il loop è attivo)
        memory = AsyncRedisSaver(redis_client=checkpointer_redis)
        
        # 2. Configura indici RedisVL (creazione fisica se non esistono)
        await memory.setup()
        
        # 3. Compila il grafo con il checkpointer
        graph = builder.compile(checkpointer=memory)
        
        logger.info("✅ Grafo compilato e Redis pronto.")
    except Exception as e:
        logger.error(f"❌ Errore durante startup: {e}")
        logger.exception(e)
    
    yield
    # Shutdown logic (opzionale)
    logger.info("👋 Shutdown server...")

app = FastAPI(title="STM32 AI Assistant API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "STM32 AI Assistant"}

@app.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    Endpoint principale per la chat.
    Riceve messaggi dall'estensione VS Code, esegue il grafo e streamma le risposte.
    """
    logger.info(f"Ricevuta richiesta chat: {len(request.messages)} messaggi")
    
    # -------------------------------------------------------------------------
    # 1. LOGICA DI INPUT E ID SESSIONE
    # -------------------------------------------------------------------------
    # user_id: Identifica CHI è l'utente (es. "michele", "michele_test").
    #          Questo ID è stabile nel tempo e permette di recuperare il PROFILO UTENTE (Long-Term Memory).
    # session_id: Identifica LA CONVERSAZIONE corrente (es. "session-001").
    #             Questo ID definisce il "Thread" di LangGraph per il checkpointer.
    #             Se cambia session_id, LangGraph parte con uno stato pulito (Short-Term Memory resettata).
    # -------------------------------------------------------------------------

    # 1. Ricostruisci input per il grafo
    # L'ultimo messaggio utente è l'input principale per il grafo
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == 'user'), "")
    
    # -------------------------------------------------------------------------
    # 2. LONG-TERM MEMORY (REDIS: user_profile)
    # -------------------------------------------------------------------------
    # Recuperiamo il "Profilo Utente" da Redis usando una chiave legata SOLO allo user_id.
    # Questa memoria sopravvive al cambio di sessione.
    # Chiave Redis: "user:{user_id}:profile"
    # -------------------------------------------------------------------------
    user_profile_key = f"user:{request.user_id}:profile"
    try:
        raw_profile = await redis_client.get(user_profile_key)
        user_profile = json.loads(raw_profile) if raw_profile else {}
        logger.info(f"👤 Profilo utente caricato per {request.user_id}: {json.dumps(user_profile, indent=2)}")
    except Exception as e:
        logger.warning(f"⚠️ Impossibile caricare profilo utente: {e}")
        user_profile = {}

    # 3. Definisci lo stato iniziale
    # Iniettiamo il profilo utente nello stato iniziale (MasterState).
    # Il grafo (nodo general_chat) userà questi dati per il recall.
    initial_state = {
        "message": last_user_message,
        "persistent_context": user_profile  # Inietta memoria a lungo termine
    }
    
    # Context injection (file aperti su VS Code, selezione, etc.)
    if request.context:
        logger.info(f"Context ricevuto: {request.context.keys()}")
    
    async def event_generator():
        try:
            # -------------------------------------------------------------------------
            # 4. SHORT-TERM MEMORY (REDIS: LangGraph Checkpoint)
            # -------------------------------------------------------------------------
            # LangGraph usa un "thread_id" per salvare lo stato di avanzamento del workflow.
            # Combinando user_id + session_id garantiamo che:
            # - Utenti diversi non si mischino.
            # - Lo stesso utente può avere più chat separate (sessioni diverse).
            # Chiave Thread: "{user_id}:{session_id}"
            # -------------------------------------------------------------------------
            composite_thread_id = f"{request.user_id}:{request.session_id}"
            config = {"configurable": {"thread_id": composite_thread_id}}

            # -------------------------------------------------------------------------
            # 4. RESUME O START: Gestione Idempotenza e Human-in-the-Loop
            # -------------------------------------------------------------------------
            current_state = await graph.aget_state(config)
            
            # Identifica se il messaggio è un comando che deve forzare il router
            msg_clean = last_user_message.lower().strip()
            triggers = [
                "firmware", "ai", "ai_analysis", "dataset", "integrazione", 
                "integration", "customization", "synthetic", "ricerca", 
                "web_research", "reset", "restart", "chiudi", "stop"
            ]
            is_workflow_trigger = (
                last_user_message.startswith("@") or 
                last_user_message.startswith("/") or 
                any(t == msg_clean for t in triggers)
            )

            if current_state.next and not is_workflow_trigger:
                logger.info(f"🔄 Resuming thread {composite_thread_id} from interrupt (waiting for: {current_state.next})")
                
                # POPULATE STATE: Inseriamo la risposta dell'utente nel campo 'user_response' dello stato.
                await graph.aupdate_state(config, {"user_response": last_user_message}) 
                
                # RESUME: Chiamiamo astream con None per proseguire dai nodi in sospeso
                stream_input = None
            else:
                if is_workflow_trigger and current_state.next:
                    logger.info(f"🚀 Detected workflow trigger '{last_user_message}': forcing RESTART and routing (ignoring pending interrupt {current_state.next})")
                
                # Forza il reset di user_response nell'input iniziale
                initial_state["user_response"] = ""
                stream_input = initial_state

            logger.info(f"▶️ Avvio esecuzione thread: {composite_thread_id}")
            
            # Eseguiamo il grafo (astream) in modalità asincrona.
            # MOTIVO ASYNC:
            # 1. Non-bloccante: Il server può gestire altri utenti mentre attende l'LLM (che è lento).
            # 2. Streaming: Possiamo inviare aggiornamenti (progress/token) al client MAN MANO che arrivano,
            #    migliorando la percezione di velocità (Time-to-First-Token basso).
            # config indica a LangGraph dove salvare lo stato intermedio.
            async for event in graph.astream(stream_input, config=config):
                
                # Cerca output dai nodi (es. "firmware_flow", "general_chat")
                for node_name, node_state in event.items():
                    logger.info(f"Nodo eseguito: {node_name}")
                    
                    # Se è un'interruzione (Human-in-the-Loop), estrai il prompt e mandala al client
                    if node_name == "__interrupt__":
                        interrupt_data = node_state[0] if isinstance(node_state, (list, tuple)) and node_state else node_state
                        value = getattr(interrupt_data, 'value', interrupt_data)
                        
                        if isinstance(value, dict) and "instruction" in value:
                            prompt_msg = f"⏸️ **AZIONE RICHIESTA**:\n\n{value['instruction']}\n\n"
                            yield json.dumps({"type": "markdown", "content": prompt_msg}) + "\n"
                        else:
                            yield json.dumps({"type": "markdown", "content": "⏸️ In attesa di input dell'utente...\n\n"}) + "\n"
                        continue

                    # Genera un messaggio di progresso JSON per il client
                    display_name = node_name.replace("_", " ").capitalize()
                    yield json.dumps({"type": "progress", "content": display_name}) + "\n"
                    
                    # Logica specifica per inviare feedback utile all'utente (Markdown)
                    if node_name == "route_request" and "route" in node_state:
                         route = node_state["route"]
                         msg = f"🔍 Ho analizzato la tua richiesta: **{route.replace('_', ' ')}**."
                         yield json.dumps({"type": "markdown", "content": f"{msg}\n\n"}) + "\n"
                    
                    # Se il nodo ha prodotto un "message" (es. chat response), invialo
                    if "message" in node_state and node_name != "route_request" and node_name != "__interrupt__":
                         yield json.dumps({"type": "markdown", "content": f"{node_state['message']}\n\n"}) + "\n"

            # 🏁 FINE PIPELINE: Salvataggio Profilo Globale (Summary). Qui il ciclo async for event in graph.astream(...) è terminato e significa che il grafo ha raggiunto il nodo END. Qui parte la logica di "salvataggio finale".
            # -------------------------------------------------------------------------
            # 5. AGGIORNAMENTO LONG-TERM MEMORY e salvataggio finale
            # -------------------------------------------------------------------------
            # Il workflow è terminato (il ciclo for è finito).
            # Ora estraiamo lo stato finale per aggiornare il Profilo Utente su Redis.
            # Questo permette alla prossima sessione di sapere cosa abbiamo fatto qui.
            # -------------------------------------------------------------------------
            try:
                final_snapshot = await graph.aget_state(config)
                state = final_snapshot.values
                is_finished = len(final_snapshot.next) == 0
                
                # ... (logica salvataggio profilo esistente) ...
                # Estrai info "importanti" da persistere
                new_profile = {
                    "board_name": state.get("board_name"),
                    "mcu_series": state.get("mcu_series"),
                    "last_workflow": state.get("route"),
                    "last_model": state.get("selected_model"),
                    "last_project_path": state.get("firmware_project_path") or state.get("firmware_project_dir"),
                    "timestamp": state.get("timestamp")
                }
                # Rimuovi None
                new_profile = {k: v for k, v in new_profile.items() if v is not None}
                
                # Unisci con il vecchio profilo
                updated_profile = {**user_profile, **new_profile}
                
                await redis_client.set(user_profile_key, json.dumps(updated_profile))
                logger.info(f"💾 Profilo aggiornato per {request.user_id}")
                
                # Invia stato finale corretto
                if is_finished:
                    yield json.dumps({"type": "status", "event": "completed", "thread_id": composite_thread_id}) + "\n"
                    yield json.dumps({"type": "markdown", "content": "✅ Elaborazione completata con successo."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "event": "waiting", "thread_id": composite_thread_id}) + "\n"
                    # Se siamo in attesa, non mandiamo il messaggio di "completata" così l'utente sa che deve rispondere
            except Exception as save_err:
                logger.warning(f"⚠️ Errore nel salvataggio del profilo o verifica finale: {save_err}")
            
        except redis.exceptions.ConnectionError:
            logger.error("❌ Errore: Impossibile connettersi a Redis. Assicurati che il container sia attivo.")
            yield json.dumps({"type": "error", "content": "Errore di connessione al database delle sessioni (Redis). Verifica che Redis sia attivo."}) + "\n"
        except Exception as e:
            logger.error(f"Errore durante esecuzione grafo: {e}")
            logger.exception(e)
            # Sanitizza messaggi per JSON
            safe_error = str(e).replace("\n", " ")
            yield json.dumps({"type": "error", "content": safe_error}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True) 

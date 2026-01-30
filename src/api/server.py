
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
    
    # 1. Ricostruisci input per il grafo
    # L'ultimo messaggio utente è l'input principale
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == 'user'), "")
    
    # Config personalizzata dal client (opzionale)
    # config = {"configurable": {"thread_id": "vscode-session"}}
    
    # 2. Recupera Profilo Globale Utente da Redis (Long-term memory)
    user_profile_key = f"user:{request.user_id}:profile"
    try:
        raw_profile = await redis_client.get(user_profile_key)
        user_profile = json.loads(raw_profile) if raw_profile else {}
        logger.info(f"👤 Profilo utente caricato per {request.user_id}: {user_profile.keys()}")
    except Exception as e:
        logger.warning(f"⚠️ Impossibile caricare profilo utente: {e}")
        user_profile = {}

    # 3. Definisci lo stato iniziale
    initial_state = {
        "message": last_user_message,
        "persistent_context": user_profile  # Inietta memoria a lungo termine
    }
    
    # Context injection (file aperti, selezione)
    if request.context:
        logger.info(f"Context ricevuto: {request.context.keys()}")
    
    async def event_generator():
        try:
            # Configurazione per LangGraph (user_id + session_id per gestione sessione)
            composite_thread_id = f"{request.user_id}:{request.session_id}"
            config = {"configurable": {"thread_id": composite_thread_id}}
            
            logger.info(f"▶️ Avvio esecuzione thread: {composite_thread_id}")
            # Nota: graph.stream restituisce eventi per ogni nodo
            async for event in graph.astream(initial_state, config=config):
                
                # Cerca output dai nodi
                for node_name, node_state in event.items():
                    logger.info(f"Nodo eseguito: {node_name}")
                    
                    # Genera un messaggio di progresso leggibile
                    display_name = node_name.replace("_", " ").capitalize()
                    yield f'{{"type": "progress", "content": "{display_name}"}}\n'
                    
                    # Logica specifica per inviare Markdown utile
                    if node_name == "route_request" and "route" in node_state:
                         route = node_state["route"]
                         msg = f"🔍 Ho analizzato la tua richiesta: **{route.replace('_', ' ')}**."
                         yield f'{{"type": "markdown", "content": "{msg}\\n\\n"}}\n'
                    
                    # Se il nodo ha dei risultati di analisi o messaggi specifici
                    if "message" in node_state and node_name != "route_request":
                         # Evitiamo di reinviare il messaggio iniziale utente
                         yield f'{{"type": "markdown", "content": "{node_state["message"]}\\n\\n"}}\n'

            # 🏁 FINE PIPELINE: Salvataggio Profilo Globale (Summary). Qui il ciclo async for event in graph.astream(...) è terminato e significa che il grafo ha raggiunto il nodo END. Qui parte la logica di "salvataggio finale".
            try:
                final_snapshot = await graph.aget_state(config)
                state = final_snapshot.values
                
                # Estrai info "importanti" da persistere
                new_profile = {
                    "board_name": state.get("board_name"),
                    "mcu_series": state.get("mcu_series"),
                    "last_workflow": state.get("route"),
                    "last_model": state.get("selected_model"),
                    "timestamp": state.get("timestamp")
                }
                # Rimuovi None
                new_profile = {k: v for k, v in new_profile.items() if v is not None}
                
                # Unisci con il vecchio profilo
                updated_profile = {**user_profile, **new_profile}
                
                await redis_client.set(user_profile_key, json.dumps(updated_profile))
                logger.info(f"💾 Profilo aggiornato per {request.user_id}")
            except Exception as save_err:
                logger.warning(f"⚠️ Errore nel salvataggio del profilo: {save_err}")

            # Risposta finale basata sull'ultimo nodo
            yield f'{{"type": "status", "event": "completed", "thread_id": "{composite_thread_id}"}}\n'
            yield f'{{"type": "markdown", "content": "✅ Elaborazione completata con successo."}}\n'
            
        except redis.exceptions.ConnectionError:
            logger.error("❌ Errore: Impossibile connettersi a Redis. Assicurati che il container sia attivo.")
            yield f'{{"type": "error", "content": "Errore di connessione al database delle sessioni (Redis). Verifica che Redis sia attivo."}}'
        except Exception as e:
            logger.error(f"Errore durante esecuzione grafo: {e}")
            logger.exception(e)
            # Sanitizza messaggi per JSON
            safe_error = str(e).replace('"', "'").replace("\n", " ")
            yield f'{{"type": "error", "content": "{safe_error}"}}\n'

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True) 

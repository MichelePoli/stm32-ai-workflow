# Questo è l'Entrypoint FastAPI Attuale (Il Cuore dell'App). È il file che viene eseguito dal Docker container langgraph-app al comando uvicorn src.api.server:app.

# * A cosa serve: Espone l'endpoint di produzione /stream. È qui che arrivano fisicamente le richieste HTTP esterne (es. dall'estensione VS Code). Riceve i messaggi e lo storico della chat, invoca l'intero Workflow intelligente (graph.py), gestisce la cache su Redis (Checkpointer) e trasforma le risposte in Server-Sent Events (SSE) per aggiornare la UI in tempo reale lettera per lettera.

# * Perché esiste: È l'API ufficiale del tuo assistente AI.

import os
import uvicorn
import logging
import redis
import json
import asyncio
from fastapi import FastAPI, Request
# from fastapi import HTTPException, Security, Depends # solo nel caso in cui implementi API KEY
from fastapi.responses import StreamingResponse
# from fastapi.security import APIKeyHeader # solo nel caso in cui implementi API KEY
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.assistant.graph import builder, redis_client, checkpointer_redis
from src.assistant.state import MasterState
from src.assistant.configuration import Configuration
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

# Configura Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_server")

# --- API Key Authentication --- (solo nel caso in cui implementi API KEY. Al momento non serve.)
# Setta API_KEY nel .env per abilitare l'autenticazione.
# Se non impostata, l'autenticazione è disabilitata (utile in sviluppo locale).
# _API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# async def verify_api_key(api_key: str = Security(_API_KEY_HEADER)):
#     expected = os.getenv("API_KEY", "")     
#     if not expected:
#         return  # Auth disabilitata: nessuna API_KEY nel .env
#     if api_key != expected:
#         raise HTTPException(status_code=403, detail="API key non valida o assente.")


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
    
    # Retry con backoff: Redis può essere in BusyLoadingError (caricamento RDB ~18s)
    import asyncio
    from redis.exceptions import BusyLoadingError
    max_retries = 20
    retry_delay = 3  # secondi
    
    for attempt in range(1, max_retries + 1):
        try:
            memory = AsyncRedisSaver(redis_client=checkpointer_redis)
            await memory.setup()
            graph = builder.compile(checkpointer=memory)
            logger.info("✅ Grafo compilato e Redis pronto.")
            break  # successo → esci dal loop
        except BusyLoadingError as e:
            if attempt < max_retries:
                logger.warning(f"⏳ Redis ancora in caricamento, riprovo in {retry_delay}s (tentativo {attempt}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Redis non pronto dopo {max_retries} tentativi: {e}")
        except Exception as e:
            logger.error(f"❌ Errore durante startup: {e}")
            logger.exception(e)
            break  # errore non recuperabile
    
    yield
    logger.info("👋 Shutdown server...")

app = FastAPI(title="STM32 AI Assistant API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "STM32 AI Assistant"}

@app.post("/stream")

# @app.post("/stream", dependencies=[Depends(verify_api_key)]) # nel caso in cui vuoi implementare API KEY (al momento no). 
async def stream_chat(request: ChatRequest):
    """
    Endpoint principale per la chat.
    Riceve messaggi dall'estensione VS Code, esegue il grafo e streamma le risposte.
    """
    logger.info(f"Ricevuta richiesta chat: {len(request.messages)} messaggi")
    
    # ultimo messaggio utente è l'input principale per il grafo
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == 'user'), "")
    
    # -------------------------------------------------------------------------
    # 2. LONG-TERM MEMORY (REDIS: user_profile)
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
    initial_state = {
        "message": last_user_message,
        "persistent_context": user_profile,
        "reset_profile": False, # Reset sticky flag
        "user_response": "",
        "response": "",
        "route": ""
    }
    
    if request.context:
        logger.info(f"Context ricevuto: {request.context.keys()}")
    
    async def event_generator():
        # Coda per aggregare eventi dal grafo e log dal subprocess
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        # Handler per catturare i log (es. training progress)
        class QueueHandler(logging.Handler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Filtriamo i log interessanti per l'utente in tempo reale
                    if any(x in msg for x in ["[Train]", "Epoch ", "accuracy:", "loss:", "[Subprocess]"]):
                        # Logica safe per spingere nella coda asincrona da un contesto sincrono
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "content": msg})
                except Exception:
                    pass

        # Collega l'handler ai logger interessati
        handler = QueueHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        loggers_to_stream = [
            logging.getLogger("src.assistant.workflow5_customization"),
            logging.getLogger("src.assistant.utils")
        ]
        for l in loggers_to_stream:
            l.addHandler(handler)

        try:
            composite_thread_id = f"{request.user_id}:{request.session_id}"
            config = {"configurable": {"thread_id": composite_thread_id}}

            current_state = await graph.aget_state(config)
            
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
                logger.info(f"🔄 Resuming thread {composite_thread_id} from interrupt")
                await graph.aupdate_state(config, {
                    "user_response": last_user_message,
                    "reset_profile": False,
                    "response": ""
                }) 
                stream_input = None
            else:
                initial_state["user_response"] = ""
                stream_input = initial_state

            # Avvia il grafo in un task separato
            async def run_graph_task():
                try:
                    async for event in graph.astream(stream_input, config=config):
                        await queue.put({"type": "graph_event", "data": event})
                except Exception as e:
                    logger.error(f"Errore nel task del grafo: {e}")
                    await queue.put({"type": "error", "content": str(e)})
                finally:
                    await queue.put(None) # Signal completion

            asyncio.create_task(run_graph_task())

            # Consuma dalla coda con heartbeat
            while True:
                try:
                    # Timeout di 15s per l'heartbeat
                    item = await asyncio.wait_for(queue.get(), timeout=15)
                    if item is None:
                        break

                    if item["type"] == "log":
                        # Stream the log to the UI
                        yield json.dumps({"type": "markdown", "content": f"_{item['content']}_\n"}) + "\n"
                    
                    elif item["type"] == "error":
                        yield json.dumps({"type": "error", "content": item["content"]}) + "\n"
                        break
                        
                    elif item["type"] == "graph_event":
                        event = item["data"]
                        for node_name, node_state in event.items():
                            logger.info(f"Nodo eseguito: {node_name}")
                            
                            if node_name == "__interrupt__":
                                interrupt_data = node_state[0] if isinstance(node_state, (list, tuple)) and node_state else node_state
                                value = getattr(interrupt_data, 'value', interrupt_data)
                                if isinstance(value, dict) and "instruction" in value:
                                    prompt_msg = f"⏸️ **AZIONE RICHIESTA**:\n\n{value['instruction']}\n\n"
                                    if "suggestion" in value:
                                        prompt_msg += f"> 💡 {value['suggestion']}\n\n"
                                    if "options" in value and isinstance(value["options"], dict):
                                        for key, text in value["options"].items():
                                            prompt_msg += f"* **{key}**: {text}\n"
                                    yield json.dumps({"type": "markdown", "content": prompt_msg}) + "\n"
                                else:
                                    yield json.dumps({"type": "markdown", "content": "⏸️ In attesa di input dell'utente...\n\n"}) + "\n"
                                continue

                            # Mappa node_name → etichetta leggibile per il progress bar
                            NODE_LABELS = {
                                # workflow 1 - firmware generation
                                "route_request": "🔀 Analisi richiesta",
                                "collect_project_info": "📋 Raccolta info progetto",
                                "search_and_install_stm32_package": "📦 Verifica package STM32",
                                "generate_cubemx_script": "📝 Generazione script CubeMX",
                                "execute_generation": "⚙️ Generazione firmware",
                                "finalize_project": "✅ Finalizzazione firmware",
                                "decide_continue_to_ai": "🔀 Decisione: analisi AI",
                                "collect_analysis_info": "📋 Raccolta info analisi AI",
                                "choose_ai_task": "🎯 Selezione task AI",
                                "choose_ai_model": "🧠 Selezione modello AI",
                                "download_model": "⬇️ Download modello",
                                "inspect_model_architecture": "🔍 Ispezione architettura",
                                "ask_modification_intent": "🛠️ Intenzione modifica modello",
                                "retrieve_best_practices_for_architecture": "📚 Best practices architettura",
                                # Workflow 5 – customization
                                "gather_user_modifications": "📝 Descrizione modifiche",
                                "ask_and_parse_user_modifications": "🧩 Parsing modifiche",
                                "collect_modification_confirmation": "✅ Conferma modifiche",
                                "apply_user_customization": "🔧 Applicazione customizzazione",
                                "ask_optimization_preference": "⚙️ Preferenza ottimizzazione",
                                "fine_tune_customized_model": "🎓 Fine-tuning modello",
                                "validate_customized_model": "✔️ Validazione modello customizzato",
                                "save_customized_model_final": "💾 Salvataggio modello finale",
                                "ask_continue_after_customization": "🔀 Continuare con analisi AI?",
                                # Workflow 6 – synthetic data
                                "ask_synthetic_data_requirements": "🧪 Requisiti dati sintetici",
                                "generate_synthetic_samples": "⚙️ Generazione dati sintetici",
                                "validate_synthetic_data": "✔️ Validazione dati sintetici",
                                # Workflow 7 – dataset
                                "decide_data_source": "🗄️ Sorgente dati",
                                "select_predefined_dataset": "📊 Selezione dataset",
                                "download_dataset": "⬇️ Download dataset",
                                # Workflow 2 – AI analysis
                                "apply_modifications": "✏️ Applicazione modifiche",
                                "run_analyze": "📊 Analisi STEdgeAI",
                                "check_resource_constraints": "⚖️ Verifica risorse MCU",
                                "run_validate": "✔️ Validazione modello",
                                "run_generate": "🏗️ Generazione codice AI",
                                "finalize_analysis": "✅ Finalizzazione analisi",
                                # Workflow 3 – integration
                                "decide_continue_to_integration": "🔀 Decisione: integrazione",
                                "collect_integration_info": "📋 Raccolta info integrazione",
                                "scan_ai_files": "🔍 Scansione file AI",
                                "copy_ai_files": "📂 Copia file AI nel firmware",
                                "modify_main_c": "✏️ Modifica main.c",
                                "verify_integration": "✔️ Verifica integrazione",
                                "finalize_integration": "✅ Finalizzazione integrazione",
                                # Workflow 4 – web search
                                "classify_search": "🔀 Classificazione ricerca",
                                "execute_web_search": "🌐 Esecuzione ricerca web",
                                "summarize_search_results": "📝 Creazione riassunto",
                                "finalize_search": "✅ Finalizzazione ricerca",
                                # General chat
                                "general_chat": "💬 Risposta chat",
                            }
                            label = NODE_LABELS.get(node_name, node_name.replace("_", " ").capitalize())
                            yield json.dumps({"type": "progress", "content": label}) + "\n"
                            
                            if node_name == "route_request" and isinstance(node_state, dict) and "route" in node_state:
                                route = node_state["route"]
                                msg = f"🔍 Ho analizzato la tua richiesta: **{route.replace('_', ' ')}**."
                                yield json.dumps({"type": "markdown", "content": f"{msg}\n\n"}) + "\n"
                            
                            # Emetti output testuale solo da nodi che producono risposte finali
                            if isinstance(node_state, dict):
                                # Risposta workflow finalizers (finalize_integration etc.)
                                if node_state.get("response"):
                                    yield json.dumps({"type": "markdown", "content": f"{node_state['response']}\n\n"}) + "\n"
                                # Chat generale: risposta salvata in state.message
                                elif node_name == "general_chat" and node_state.get("message"):
                                    yield json.dumps({"type": "markdown", "content": f"{node_state['message']}\n\n"}) + "\n"
                                # Firmware finalizer
                                elif node_name == "finalize_project" and node_state.get("firmware_project_path"):
                                    path = node_state["firmware_project_path"]
                                    yield json.dumps({"type": "markdown", "content": f"✓ Progetto firmware generato: `{path}`\n\n"}) + "\n"
                                # AI analysis finalizer
                                elif node_name == "finalize_analysis" and node_state.get("ai_code_dir"):
                                    yield json.dumps({"type": "markdown", "content": f"✓ Analisi AI completata! Codice generato in: `{node_state['ai_code_dir']}`\n\n"}) + "\n"


                
                except asyncio.TimeoutError:
                    # Heartbeat: manda un pacchetto vuoto o un progress silenzioso
                    yield json.dumps({"type": "progress", "content": "..."}) + "\n"

            # Logica di salvataggio finale profilo (dopo fine coda)
            try:
                final_snapshot = await graph.aget_state(config)
                state = final_snapshot.values
                is_finished = len(final_snapshot.next) == 0
                
                # Estraiamo i valori correnti dallo stato
                new_profile = {
                    "board_name": state.get("board_name"),
                    "mcu_series": state.get("mcu_series"),
                    "last_model": state.get("selected_model"),
                    "last_project_path": state.get("firmware_project_path") or state.get("firmware_project_dir"),
                    "last_workflow": state.get("route"),
                    "timestamp": state.get("timestamp")
                }
                
                # Rimuovi solo i None e le stringhe vuote, ma mantieni tutto il resto (incluso F401)
                new_profile = {k: v for k, v in new_profile.items() if v is not None and v != ""}
                
                # Unisci con il profilo esistente (nuovi valori vincono)
                updated_profile = {**user_profile, **new_profile}
                
                if state.get("reset_profile"):
                    updated_profile = {}
                
                await redis_client.set(user_profile_key, json.dumps(updated_profile))
                logger.info(f"💾 Profilo salvato per {request.user_id}: {json.dumps(updated_profile)}")
                
                if is_finished:
                    yield json.dumps({"type": "status", "event": "completed", "thread_id": composite_thread_id}) + "\n"
                    yield json.dumps({"type": "markdown", "content": "✅ Elaborazione completata con successo."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "event": "waiting", "thread_id": composite_thread_id}) + "\n"
            except Exception as se:
                logger.warning(f"⚠️ Errore salvataggio profilo: {se}")

        finally:
            for l in loggers_to_stream:
                l.removeHandler(handler)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)

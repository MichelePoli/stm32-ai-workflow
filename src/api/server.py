
import os
import uvicorn
import logging
import redis
import json
import asyncio
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
        "persistent_context": user_profile
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
                await graph.aupdate_state(config, {"user_response": last_user_message}) 
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
                                    yield json.dumps({"type": "markdown", "content": prompt_msg}) + "\n"
                                else:
                                    yield json.dumps({"type": "markdown", "content": "⏸️ In attesa di input dell'utente...\n\n"}) + "\n"
                                continue

                            display_name = node_name.replace("_", " ").capitalize()
                            yield json.dumps({"type": "progress", "content": display_name}) + "\n"
                            
                            if node_name == "route_request" and "route" in node_state:
                                 route = node_state["route"]
                                 msg = f"🔍 Ho analizzato la tua richiesta: **{route.replace('_', ' ')}**."
                                 yield json.dumps({"type": "markdown", "content": f"{msg}\n\n"}) + "\n"
                            
                            if "message" in node_state and node_name != "route_request" and node_name != "__interrupt__":
                                 yield json.dumps({"type": "markdown", "content": f"{node_state['message']}\n\n"}) + "\n"
                
                except asyncio.TimeoutError:
                    # Heartbeat: manda un pacchetto vuoto o un progress silenzioso
                    yield json.dumps({"type": "progress", "content": "..."}) + "\n"

            # Logica di salvataggio finale profilo (dopo fine coda)
            try:
                final_snapshot = await graph.aget_state(config)
                state = final_snapshot.values
                is_finished = len(final_snapshot.next) == 0
                
                new_profile = {
                    "board_name": state.get("board_name"),
                    "mcu_series": state.get("mcu_series"),
                    "last_workflow": state.get("route"),
                    "last_model": state.get("selected_model"),
                    "last_project_path": state.get("firmware_project_path") or state.get("firmware_project_dir"),
                    "timestamp": state.get("timestamp")
                }
                new_profile = {k: v for k, v in new_profile.items() if v is not None}
                updated_profile = {**user_profile, **new_profile}
                
                if state.get("reset_profile"):
                    updated_profile = {}
                
                await redis_client.set(user_profile_key, json.dumps(updated_profile))
                
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

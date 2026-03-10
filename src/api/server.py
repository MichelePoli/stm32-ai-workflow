# Questo è l'Entrypoint FastAPI Attuale (Il Cuore dell'App). È il file che viene eseguito dal Docker container langgraph-app al comando uvicorn src.api.server:app.

# * What it does: Exposes the production endpoint /stream. This is where external HTTP requests physically arrive (e.g., from the VS Code extension). It receives messages and chat history, invokes the entire Intelligent Workflow (graph.py), manages the Redis cache (Checkpointer), and transforms responses into Server-Sent Events (SSE) to update the UI in real time letter by letter.

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

# Configure Logging
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
#         raise HTTPException(status_code=403, detail="Invalid or missing API key.")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "default-session"

# Global placeholders for the graph (initialized on startup)
graph = None
memory = None

def format_sse(data: str) -> str:
    """Format string for Server-Sent Events (optional if client handles raw chunks)"""
    return f"data: {data}\n\n"

from contextlib import asynccontextmanager


async def unload_all_triton_models() -> None:
    """
    Sends an unload request to Triton for every known heavy model at startup.

    This guarantees a clean VRAM state regardless of what was left loaded by
    a previous session. Each call is fire-and-forget: failures are logged as
    warnings so they never block the server from booting.
    """
    import urllib.request
    from urllib.error import HTTPError

    triton_base = os.getenv("TRITON_BASE_URL", "http://localhost:8000/v1")
    # Strip the /v1 suffix to reach the native Triton v2 repository API
    triton_base = triton_base.rstrip("/").removesuffix("/v1")

    # All heavy LLMs managed via explicit model control.
    # nomic-embed is intentionally excluded: it is small and always needed.
    models_to_unload = ["mistral", "deepseek-r1", "gpt-oss-20b"]

    logger.info("🧹 [Startup] Unloading all Triton models to reset VRAM...")
    for model in models_to_unload:
        url = f"{triton_base}/v2/repository/models/{model}/unload"
        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req, timeout=10) as _:
                pass
            logger.info(f"   ✅ Unloaded: {model}")
        except HTTPError as e:
            if e.code == 400:
                # Model was not loaded — this is the normal case at first boot.
                logger.info(f"   ℹ️  {model}: not loaded (skipped)")
            else:
                logger.warning(f"   ⚠️  {model}: HTTP {e.code} during unload — {e}")
        except Exception as e:
            logger.warning(f"   ⚠️  {model}: unload failed — {e}")
    logger.info("🧹 [Startup] VRAM reset complete.")


# Graph Initialization & Redis Checkpointer (lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the graph and Redis checkpointer on startup (inside the event loop)."""
    global graph, memory
    logger.info("🚀 Initializing Graph & Redis Checkpointer...")
    
    # Retry with backoff: Redis might be in BusyLoadingError (RDB load ~18s)
    import asyncio
    from redis.exceptions import BusyLoadingError
    max_retries = 20
    retry_delay = 3  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            memory = AsyncRedisSaver(redis_client=checkpointer_redis)
            await memory.setup()
            graph = builder.compile(checkpointer=memory)
            logger.info("✅ Graph compiled and Redis ready.")
            # Unload all Triton models to start from a clean VRAM state
            # await unload_all_triton_models() # only for debug, then comment this line 
            break  # success -> exit loop 
        except BusyLoadingError as e:
            if attempt < max_retries:
                logger.warning(f"⏳ Redis is still loading, retrying in {retry_delay}s (attempt {attempt}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Redis not ready after {max_retries} attempts: {e}")
        except Exception as e:
            logger.error(f"❌ Error during startup: {e}")
            logger.exception(e)
            break  # unrecoverable error
    
    yield
    logger.info("👋 Shutting down server...")

app = FastAPI(title="STM32 AI Assistant API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "STM32 AI Assistant"}

@app.post("/stream")

# @app.post("/stream", dependencies=[Depends(verify_api_key)]) # nel caso in cui vuoi implementare API KEY (al momento no). 
async def stream_chat(request: ChatRequest):
    """
    Main endpoint for chat.
    Receives messages from the VS Code extension, executes the graph and streams the responses.
    """
    logger.info(f"Received chat request: {len(request.messages)} messages")
    
    # last user message is the main input for the graph
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == 'user'), "")
    
    # -------------------------------------------------------------------------
    # 2. LONG-TERM MEMORY (REDIS: user_profile)
    # -------------------------------------------------------------------------
    user_profile_key = f"user:{request.user_id}:profile"
    try:
        raw_profile = await redis_client.get(user_profile_key)
        user_profile = json.loads(raw_profile) if raw_profile else {}
        logger.info(f"👤 User profile loaded for {request.user_id}: {json.dumps(user_profile, indent=2)}")
    except Exception as e:
        logger.warning(f"⚠️ Unable to load user profile: {e}")
        user_profile = {}

    # 3. Define the initial state
    initial_state = {
        "message": last_user_message,
        "persistent_context": user_profile,
        "reset_profile": False, # Reset sticky flag
        "user_response": "",
        "response": "",
        "route": ""
    }
    
    if request.context:
        logger.info(f"Context received: {request.context.keys()}")
    
    async def event_generator():
        # Queue to aggregate events from the graph and logs from subprocesses
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        
        composite_thread_id = f"{request.user_id}:{request.session_id}"

        # Handler to capture logs (e.g. training progress)
        class QueueHandler(logging.Handler):
            def __init__(self, target_thread):
                super().__init__()
                self.target_thread = target_thread

            def emit(self, record):
                try:
                    # Filter strictly by thread_id if the log comes from a specific thread's subprocess
                    if hasattr(record, "thread_id") and record.thread_id != self.target_thread:
                        return
                    
                    msg = self.format(record)
                    # We filter interesting logs for the user in real-time
                    if any(x in msg for x in ["[Train]", "Epoch ", "accuracy:", "loss:", "[Subprocess]"]):
                        # Safe logic to push into the asynchronous queue from a synchronous context
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "log", "content": msg})
                except Exception:
                    pass

        # Connect the handler to the interested loggers
        handler = QueueHandler(target_thread=composite_thread_id)
        handler.setFormatter(logging.Formatter('%(message)s'))
        loggers_to_stream = [
            logging.getLogger("src.assistant.workflow5_customization"),
            logging.getLogger("src.assistant.utils")
        ]
        for l in loggers_to_stream:
            l.addHandler(handler)

        try:
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

            # Execute the graph in a separate task
            async def run_graph_task():
                try:
                    async for event in graph.astream(stream_input, config=config):
                        await queue.put({"type": "graph_event", "data": event})
                except Exception as e:
                    logger.error(f"Error in graph task: {e}")
                    await queue.put({"type": "error", "content": str(e)})
                finally:
                    await queue.put(None) # Signal completion

            asyncio.create_task(run_graph_task())

            # Consume from the queue with heartbeat
            while True:
                try:
                    # 15s timeout for heartbeat
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
                            logger.info(f"Node executed: {node_name}")
                            
                            if node_name == "__interrupt__":
                                interrupt_data = node_state[0] if isinstance(node_state, (list, tuple)) and node_state else node_state
                                value = getattr(interrupt_data, 'value', interrupt_data)
                                if isinstance(value, dict) and "instruction" in value:
                                    prompt_msg = f"⏸️ **ACTION REQUIRED**:\n\n{value['instruction']}\n\n"
                                    if "suggestion" in value:
                                        prompt_msg += f"> 💡 {value['suggestion']}\n\n"
                                    if "options" in value and isinstance(value["options"], dict):
                                        for key, text in value["options"].items():
                                            prompt_msg += f"* **{key}**: {text}\n"
                                    yield json.dumps({"type": "markdown", "content": prompt_msg}) + "\n"
                                else:
                                    yield json.dumps({"type": "markdown", "content": "⏸️ Waiting for user input...\n\n"}) + "\n"
                                continue

                            # Map node_name → readable label for the progress bar
                            NODE_LABELS = {
                                # workflow 1 - firmware generation
                                "route_request": "🔀 Request Analysis",
                                "collect_project_info": "📋 Collecting Project Info",
                                "search_and_install_stm32_package": "📦 Checking STM32 Package",
                                "generate_cubemx_script": "📝 Generating CubeMX Script",
                                "execute_generation": "⚙️ Generating Firmware",
                                "finalize_project": "✅ Finalizing Firmware",
                                "decide_continue_to_ai": "🔀 Decision: AI Analysis",
                                "collect_analysis_info": "📋 Collecting AI Analysis Info",
                                "choose_ai_task": "🎯 Selecting AI Task",
                                "choose_ai_model": "🧠 Selecting AI Model",
                                "download_model": "⬇️ Downloading Model",
                                "inspect_model_architecture": "🔍 Inspecting Architecture",
                                "ask_modification_intent": "🛠️ Model Modification Intent",
                                "retrieve_best_practices_for_architecture": "📚 Architecture Best Practices",
                                # Workflow 5 - customization
                                "gather_user_modifications": "📝 Modifications Description",
                                "ask_and_parse_user_modifications": "🧩 Parsing Modifications",
                                "collect_modification_confirmation": "✅ Confirming Modifications",
                                "apply_user_customization": "🔧 Applying Customization",
                                "ask_optimization_preference": "⚙️ Optimization Preference",
                                "optimize_hyperparameters_with_nni": "🧪 Optimize hyperparameters with NNI",
                                "fine_tune_customized_model": "🎓 Fine-tuning Model",
                                "validate_customized_model": "✔️ Validating Customized Model",
                                "save_customized_model_final": "💾 Saving Final Model",
                                "ask_continue_after_customization": "🔀 Continue with AI Analysis?",
                                # Workflow 6 - synthetic data
                                "ask_synthetic_data_requirements": "🧪 Synthetic Data Requirements",
                                "generate_synthetic_samples": "⚙️ Generating Synthetic Data",
                                "validate_synthetic_data": "✔️ Validating Synthetic Data",
                                # Workflow 7 - dataset
                                "decide_data_source": "🗄️ Data Source",
                                "select_predefined_dataset": "📊 Selecting Dataset",
                                "download_dataset": "⬇️ Downloading Dataset",
                                # Workflow 2 - AI analysis
                                "apply_modifications": "✏️ Applying Modifications",
                                "run_analyze": "📊 STEdgeAI Analysis",
                                "check_resource_constraints": "⚖️ Checking MCU Resources",
                                "run_validate": "✔️ Validating Model",
                                "run_generate": "🏗️ Generating AI Code",
                                "finalize_analysis": "✅ Finalizing Analysis",
                                # Workflow 3 - integration
                                "decide_continue_to_integration": "🔀 Decision: Integration",
                                "collect_integration_info": "📋 Collecting Integration Info",
                                "scan_ai_files": "🔍 Scanning AI Files",
                                "copy_ai_files": "📂 Copying AI files to Firmware",
                                "modify_main_c": "✏️ Modifying main.c",
                                "verify_integration": "✔️ Verifying Integration",
                                "finalize_integration": "✅ Finalizing Integration",
                                # Workflow 4 - web search
                                "classify_search": "🔀 Classifying Search",
                                "execute_web_search": "🌐 Executing Web Search",
                                "summarize_search_results": "📝 Creating Summary",
                                "finalize_search": "✅ Finalizing Search",
                                # General chat
                                "general_chat": "💬 Chat Response",
                            }
                            label = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                            yield json.dumps({"type": "progress", "content": label}) + "\n"
                            
                            if node_name == "route_request" and isinstance(node_state, dict) and "route" in node_state:
                                route = node_state["route"]
                                msg = f"🔍 I have analyzed your request: **{route.replace('_', ' ')}**."
                                yield json.dumps({"type": "markdown", "content": f"{msg}\n\n"}) + "\n"
                            
                            # Emit textual output only from nodes that produce final responses
                            if isinstance(node_state, dict):
                                # Response workflow finalizers (finalize_integration etc.)
                                if node_state.get("response"):
                                    yield json.dumps({"type": "markdown", "content": f"{node_state['response']}\n\n"}) + "\n"
                                # General chat: response saved in state.message
                                elif node_name == "general_chat" and node_state.get("message"):
                                    yield json.dumps({"type": "markdown", "content": f"{node_state['message']}\n\n"}) + "\n"
                                # Firmware finalizer
                                elif node_name == "finalize_project" and node_state.get("firmware_project_path"):
                                    path = node_state["firmware_project_path"]
                                    yield json.dumps({"type": "markdown", "content": f"✓ Firmware project generated: `{path}`\n\n"}) + "\n"
                                # AI analysis finalizer
                                elif node_name == "finalize_analysis" and node_state.get("ai_code_dir"):
                                    yield json.dumps({"type": "markdown", "content": f"✓ AI Analysis completed! Code generated in: `{node_state['ai_code_dir']}`\n\n"}) + "\n"


                
                except asyncio.TimeoutError:
                    # Heartbeat: send an empty packet or a silent progress
                    yield json.dumps({"type": "progress", "content": "..."}) + "\n"

            # Final profile save logic (after queue ends)
            try:
                final_snapshot = await graph.aget_state(config)
                state = final_snapshot.values
                is_finished = len(final_snapshot.next) == 0
                
                # We extract current values from state
                new_profile = {
                    "board_name": state.get("board_name"),
                    "mcu_series": state.get("mcu_series"),
                    "last_model": state.get("selected_model"),
                    "last_project_path": state.get("firmware_project_path") or state.get("firmware_project_dir"),
                    "last_workflow": state.get("route"),
                    "timestamp": state.get("timestamp")
                }
                
                # Remove only None and empty strings, but keep everything else (including F401)
                new_profile = {k: v for k, v in new_profile.items() if v is not None and v != ""}
                
                # Merge with the existing profile (new values win)
                updated_profile = {**user_profile, **new_profile}
                
                if state.get("reset_profile"):
                    updated_profile = {}
                
                await redis_client.set(user_profile_key, json.dumps(updated_profile))
                logger.info(f"💾 Profile saved for {request.user_id}: {json.dumps(updated_profile)}")
                
                if is_finished:
                    yield json.dumps({"type": "status", "event": "completed", "thread_id": composite_thread_id}) + "\n"
                    yield json.dumps({"type": "markdown", "content": "✅ Processing completed successfully."}) + "\n"
                else:
                    yield json.dumps({"type": "status", "event": "waiting", "thread_id": composite_thread_id}) + "\n"
            except Exception as se:
                logger.warning(f"⚠️ Error saving profile: {se}")

        finally:
            for l in loggers_to_stream:
                l.removeHandler(handler)

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="127.0.0.1", port=8000, reload=True)

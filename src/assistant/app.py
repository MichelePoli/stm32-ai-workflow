import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Import graph application from src/assistant/graph.py
from src.assistant.graph import graph
from src.assistant.state import MasterInput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="STM32 AI Assistant API", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    user_response: Optional[str] = ""
    # Optional context (can be passed from client for state restoration if needed)
    persistent_context: Optional[Dict[str, Any]] = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Endpoint to interact with the LangGraph assistant.
    Currently stateless per request unless using checkpointer (not configured here).
    """
    try:
        logger.info(f"Received chat request: {req.message[:50]}...")
        
        # Construct input dict matching MasterInput TypedDict
        initial_state: MasterInput = {
            "message": req.message,
            "user_response": req.user_response or "",
            "persistent_context": req.persistent_context or {}
        }
        
        # Invoke the graph asynchronously
        # Using ainvoke allows better concurrency handling
        result = await graph.ainvoke(initial_state)
        
        # Extract the final message from the state
        response_text = result.get("message", "No response generated.")
        route = result.get("route", "unknown")
        
        return {
            "response": response_text,
            "route": route,
            # Ideally we'd return a thread_id or similar if using checkpointer
        }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

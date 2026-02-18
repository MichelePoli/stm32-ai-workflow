import logging
import json
import urllib.request
from typing import Any, Dict, List, Optional, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from openai import OpenAI

logger = logging.getLogger(__name__)

class ChatTriton(BaseChatModel):
    """
    LangChain wrapper for Nvidia Triton Inference Server (via OpenAI-compatible API).
    Assuming Triton is running vLLM or similar python wrapper that exposes /v1/chat/completions.
    """
    client: Any = None
    model_name: str = "mistral"
    triton_url: str = "http://localhost:8000/v1"
    temperature: float = 0.7
    max_tokens: int = 2048

    def __init__(self, triton_url: str, model: str, temperature: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.triton_url = triton_url
        self.model_name = model
        self.temperature = temperature
        # Initialize OpenAI client pointing to Triton
        self.client = OpenAI(
            base_url=self.triton_url,
            api_key="empty" # Triton/vLLM usually doesn't need a key locally
        )

    @property
    def _llm_type(self) -> str:
        return "triton-chat-openai-wrapper"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        openai_messages = self._convert_messages(messages)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=False
        )
        
        content = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        openai_messages = self._convert_messages(messages)
        
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield ChatGenerationChunk(message=AIMessage(content=content))
                if run_manager:
                    run_manager.on_llm_new_token(content)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        openai_msgs = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage): role = "system"
            elif isinstance(m, AIMessage): role = "assistant"
            openai_msgs.append({"role": role, "content": m.content})
        return openai_msgs


class TritonEmbeddings:
    """
    LangChain-compatible wrapper for Triton Inference Server embeddings.
    
    Uses Triton's native HTTP inference API (POST /v2/models/{model}/infer)
    instead of the OpenAI /v1/embeddings endpoint, which is NOT exposed
    by Triton's Python backend by default.
    """
    def __init__(self, triton_url: str, model_name: str = "nomic-embed"):
        # triton_url is like "http://triton-server:8000/v1"
        # We need the base without /v1 for the native Triton API
        self.base_url = triton_url.rstrip("/").removesuffix("/v1")
        self.model_name = model_name

    def _infer(self, text: str) -> List[float]:
        """Call Triton's native HTTP inference endpoint."""
        url = f"{self.base_url}/v2/models/{self.model_name}/infer"
        payload = {
            "inputs": [
                {
                    "name": "TEXT",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [text]
                }
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read())
            # Extract the embedding from the Triton response
            return result["outputs"][0]["data"]
        except Exception as e:
            logger.error(f"❌ Triton embedding inference failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._infer(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._infer(text)

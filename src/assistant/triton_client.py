import logging
import json
import urllib.request
from typing import Any, Dict, List, Optional, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk

from openai import OpenAI

from langchain_core.runnables import RunnableSerializable, RunnableConfig, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

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

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableSerializable:
        """
        Supporta l'output strutturato forzando il modello a restituire JSON.
        Nota: Triton/vLLM non sempre supportano 'response_format={'type': 'json_object'}',
        quindi usiamo un prompt rinforzato e parsing manuale.
        """
        from langchain_core.output_parsers import JsonOutputParser
        parser = JsonOutputParser(pydantic_object=schema)

        def _format_prompt(messages: List[BaseMessage]) -> List[BaseMessage]:
            # Aggiungi le istruzioni del parser al messaggio di sistema
            if isinstance(messages[0], SystemMessage):
                messages[0].content += f"\n\nReturn ONLY a valid JSON object matching this schema:\n{parser.get_format_instructions()}"
            else:
                messages.insert(0, SystemMessage(content=f"Return ONLY a valid JSON object matching this schema:\n{parser.get_format_instructions()}"))
            return messages

        chain = RunnableLambda(_format_prompt) | self | parser
        return chain

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        openai_messages = self._convert_messages(messages)
        
        # Ensure model is loaded in EXPLICIT mode
        self._ensure_model_loaded()
        
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
        
        # Ensure model is loaded in EXPLICIT mode
        self._ensure_model_loaded()
        
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

    def _ensure_model_loaded(self) -> None:
        """
        Garantisce che il modello sia caricato in Triton (modalità EXPLICIT).
        Gestisce lo swapping mutualmente esclusivo per non superare i 16GB di VRAM.
        Su A4000 (16GB), manteniamo solo UN modello LLM attivo alla volta.
        """
        base_url = self.triton_url.rstrip("/").removesuffix("/v1")
        
        # Lista di tutti i modelli LLM (escludendo nomic-embed che è piccolo)
        all_llms = ["mistral", "deepseek-r1", "gpt-oss-20b"]
        
        # 1. Verifica se il modello richiesto è già pronto
        if self._is_model_ready(base_url, self.model_name):
            return

        # 2. Unload aggressivo di TUTTI gli altri LLM e attesa sincronizzazione
        for model in all_llms:
            if model != self.model_name:
                # Controlliamo se è caricato o in fase di caricamento
                if self._check_model_status(base_url, model) != "UNAVAILABLE":
                    logger.info(f"⏳ Scaricamento modello {model} per liberare VRAM...")
                    self._unload_model(base_url, model)
                    self._wait_for_status(base_url, model, "UNAVAILABLE")
                    # Sleep di sicurezza per permettere al backend Python di chiudere il processo
                    import time
                    time.sleep(2)

        # 3. Richiesta di caricamento per il modello target
        logger.info(f"⏳ Caricamento modello target: {self.model_name}...")
        url = f"{base_url}/v2/repository/models/{self.model_name}/load"
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pass
            # Attendiamo che diventi effettivamente READY
            self._wait_for_status(base_url, self.model_name, "READY", timeout=180)
            logger.info(f"✅ Modello {self.model_name} caricato con successo.")
        except Exception as e:
            logger.error(f"❌ Errore durante caricamento Triton: {e}")

    def _check_model_status(self, base_url: str, model_name: str) -> str:
        """Controlla lo stato del modello tramite repository API."""
        url = f"{base_url}/v2/repository/index"
        payload = {"name": model_name}
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                repo_index = json.loads(response.read())
                for m in repo_index:
                    if m["name"] == model_name:
                        return m.get("state", "UNAVAILABLE")
            return "UNAVAILABLE"
        except Exception:
            return "UNAVAILABLE"

    def _wait_for_status(self, base_url: str, model_name: str, target_state: str, timeout: int = 120) -> None:
        """Polling dello stato del modello."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            current = self._check_model_status(base_url, model_name)
            if current == target_state:
                return
            if target_state == "UNAVAILABLE" and current == "UNAVAILABLE":
                return
            time.sleep(2)
        logger.warning(f"⚠️ Timeout attendendo stato {target_state} per {model_name}")

    def _is_model_ready(self, base_url: str, model_name: str) -> bool:
        """Shorthand per verificare se un modello è READY."""
        return self._check_model_status(base_url, model_name) == "READY"

    def _unload_model(self, base_url: str, model_to_unload: str) -> None:
        """Invia comando di scaricamento."""
        url = f"{base_url}/v2/repository/models/{model_to_unload}/unload"
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                pass
            logger.info(f"⬇️ Unloaded {model_to_unload} to free VRAM")
        except Exception:
            pass


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

    def _ensure_model_loaded(self) -> None:
        """Call Triton's repository API to ensure the model is loaded."""
        url = f"{self.base_url}/v2/repository/models/{self.model_name}/load"
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                pass
            logger.info(f"✅ Model {self.model_name} is loaded (or loading started)")
        except Exception as e:
            logger.debug(f"ℹ️ Triton model load call (optional) for {self.model_name}: {e}")

    def _infer(self, text: str) -> List[float]:
        """Call Triton's native HTTP inference endpoint."""
        self._ensure_model_loaded()
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

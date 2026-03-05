# Questo è il codice che gira dentro l'App Python (LangGraph container). È il "traduttore" che permette all'assistente AI di parlare con Triton.

# * A cosa serve: Espone interfacce compatibili con LangChain (es. ChatTritonVLLM), permettendo al tuo workflow di fare richieste ai modelli usando una sintassi AI standard. Gestisce internamente le chiamate di rete (HTTP/gRPC) verso la porta 8000 di Triton, si occupa di caricare/scaricare i modelli dalla VRAM automagicamente tramite API, ed estrae il testo generato.

# * Perché esiste: LangGraph si aspetta di parlare con "OpenAI" o "Anthropic". Questo file inganna LangGraph facendogli credere che Triton sia una normale API AI.

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
    LangChain wrapper for Nvidia Triton Inference Server.
    
    This class bridges the gap between LangChain's standardized interface and Triton's 
    Specialized Inference API. It handles model lifecycle management (load/unload)
    to operate within limited VRAM constraints (16GB on NVIDIA A4000).
    """
    client: Any = None
    model_name: str = "mistral"
    triton_url: str = "http://localhost:8000/v1"
    temperature: float = 0.7
    max_tokens: int = 2048
    stop_sequences: List[str] = []

    def __init__(self, triton_url: str, model: str, temperature: float = 0.7, stop: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.triton_url = triton_url
        self.model_name = model
        self.temperature = temperature
        self.stop_sequences = stop or []

        # Initialize OpenAI client pointing to Triton
        self.client = OpenAI(
            base_url=self.triton_url,
            api_key="empty" # Triton/vLLM usually doesn't need a key locally
        )

    @property
    def base_v2_url(self) -> str:
        """
        Returns the base URL for Triton V2 APIs (repository, inference).
        Strips the /v1 suffix if present to handle proxies and prefixes (e.g. /triton/v1).
        """
        return self.triton_url.rstrip("/").removesuffix("/v1")

    @property
    def _llm_type(self) -> str:
        return "triton-chat-openai-wrapper"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableSerializable:
        """
        Enables structured output by forcing the model to return JSON.
        
        This method:
        1. Injects JSON schema instructions into the system prompt.
        2. Routes the request through the LLM.
        3. Extracts the JSON block from the potentially noisy string response.
        4. Validates and parses the JSON into the requested Pydantic schema.
        """
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda
        import re

        parser = JsonOutputParser(pydantic_object=schema)

        def _inject_schema(messages: List[BaseMessage]) -> List[BaseMessage]:
            """Inject JSON schema instructions into the system message.
            Keep it short — ottimizzazione valida (prompt più corto = meno rischio di timeout vuoti), ma non era la causa root.
            """
            # Minimal instruction: just list field names from the schema
            if hasattr(schema, 'model_fields'):
                field_names = list(schema.model_fields.keys())
                schema_hint = f"Required JSON fields: {field_names}"
            else:
                schema_hint = parser.get_format_instructions()[:400]  # truncate

            schema_instructions = (
                f"\n\nRespond ONLY with a valid JSON object. No markdown. No explanation. {schema_hint}"
            )
            messages = list(messages)  # don't mutate caller's list
            if messages and isinstance(messages[0], SystemMessage):
                messages[0] = SystemMessage(content=messages[0].content + schema_instructions)
            else:
                messages.insert(0, SystemMessage(content=schema_instructions))
            return messages

        def _extract_json(ai_message) -> str:
            """Find and extract the first JSON object from the LLM response.
            
            Returns '{}' (empty object) instead of empty string when Mistral
            returns nothing — this prevents JsonOutputParser from crashing.
            """
            text = ai_message.content if hasattr(ai_message, 'content') else str(ai_message)
            
            if not text or not text.strip():
                logger.warning("⚠️ Mistral returned empty response for structured output, using fallback {}")
                return '{}'

            # Find first { and last }
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
                
            # Fallback to fence stripping logic for non-dict JSON (if ever used)
            text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text.strip())
            
            # If still no braces, return empty object so parser doesn't crash
            if not text.strip():
                return '{}'
            return text


        def _to_pydantic(data: dict) -> Any:
            """Convert parsed dict to Pydantic model instance for attribute access.
            
            Falls back to returning the raw dict if validation fails (e.g. Mistral returned
            an incomplete / empty JSON). This prevents a ValidationError from crashing
            inside the chain — the caller's try/except block will handle it instead.
            """
            if isinstance(data, dict) and hasattr(schema, 'model_validate'):
                try:
                    return schema.model_validate(data)
                except Exception as ve:
                    logger.warning(
                        f"⚠️ with_structured_output: Pydantic validation failed for {schema.__name__ if hasattr(schema, '__name__') else schema} "
                        f"(LLM returned incomplete JSON). Raw dict returned for caller to handle. Error: {ve}"
                    )
                    return data  # return raw dict; caller's try/except will deal with it
            return data

        chain = (
            RunnableLambda(_inject_schema)
            | self
            | RunnableLambda(_extract_json)
            | parser
            | RunnableLambda(_to_pydantic)
        )
        return chain



    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        # 1. ORCHESTRATE VRAM: Ensure the target model is loaded in Triton's memory.
        # Since Triton is in EXPLICIT mode, it won't load the model until requested.
        self._ensure_model_loaded()
        
        # 2. PROMPT FORMATTING: Triton's Python backend using vLLM expects a raw text 
        # string in the "PROMPT" input tensor. We collapse the list of chat messages 
        # (System, Human, AI) into a single formatted string.
        prompt = self._format_prompt(messages)
        
        # Native Triton v2 inference with retry for server-side transient errors
        # Native Triton v2 inference with retry for server-side transient errors
        infer_url = f"{self.base_v2_url}/v2/models/{self.model_name}/infer"
        payload = {
            "inputs": [
                {
                    "name": "PROMPT",
                    "shape": [1, 1],
                    "datatype": "BYTES",
                    "data": [prompt]
                }
            ]
        }
        
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    infer_url,
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120) as response:
                    res_body = json.loads(response.read())
                    output_data = res_body["outputs"][0]["data"][0]
                    content = output_data if isinstance(output_data, str) else output_data.decode("utf-8")
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    logger.warning(f"⚠️ Inference attempt {attempt+1}/{max_retries} failed ({e}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # Triton Python backend does not support true streaming.
        # Fall back to full generation and yield the result as a single chunk.
        result = self._generate(messages, stop, run_manager, **kwargs)
        content = result.generations[0].message.content
        yield ChatGenerationChunk(message=AIMessage(content=content))
        if run_manager:
            run_manager.on_llm_new_token(content)


    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert to OpenAI-style message dicts (kept for with_structured_output path)."""
        openai_msgs = []
        for m in messages:
            role = "user"
            if isinstance(m, SystemMessage): role = "system"
            elif isinstance(m, AIMessage): role = "assistant"
            openai_msgs.append({"role": role, "content": m.content})
        return openai_msgs

    def _format_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Format a list of LangChain messages into a single prompt string
        suitable for the Triton Python backend PROMPT tensor.
        """
        prompt = ""
        for m in messages:
            if isinstance(m, SystemMessage):
                prompt += f"System: {m.content}\n"
            elif isinstance(m, HumanMessage):
                prompt += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                prompt += f"Assistant: {m.content}\n"
        prompt += "Assistant: "
        return prompt

    def _ensure_model_loaded(self) -> None:
        """
        Guarantees that the requested model is READY in Triton (EXPLICIT mode).

        On the HPP server there is sufficient VRAM to keep all LLMs loaded
        simultaneously, so no model-swapping / mutual-exclusion is required.
        The method simply checks whether the target model is already READY; if
        not, it issues a load request and waits for the endpoint to become live.
        """
        base_url = self.base_v2_url

        # Fast-path: model already READY, nothing to do.
        if self._is_model_ready(base_url, self.model_name):
            return

        # Load the target model and wait until it is fully ready.
        logger.info(f"⏳ Caricamento modello target: {self.model_name}...")
        url = f"{self.base_v2_url}/v2/repository/models/{self.model_name}/load"
        req = urllib.request.Request(url, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                pass
            # Wait until the model appears as READY in the repository index.
            self._wait_for_status(base_url, self.model_name, "READY", timeout=180)
            # Probe the native v2 infer endpoint: a READY repo state does NOT
            # guarantee the HTTP endpoint is registered yet (CUDA graph capturing
            # can finish *after* the status flip).
            self._wait_for_endpoint_live(timeout=120)
            logger.info(f"✅ Modello {self.model_name} caricato e endpoint live.")
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

    def _wait_for_endpoint_live(self, timeout: int = 60) -> None:
        """
        Probe the native Triton v2 infer endpoint until it stops returning errors.
        This bridges the gap between Triton's repository state flipping to READY
        and the model's execute() method actually being ready to handle requests
        (the Python backend stub needs a moment to become fully operational).
        """
        import time

        base_url = self.triton_url.rstrip("/").removesuffix("/v1")
        infer_url = f"{base_url}/v2/models/{self.model_name}/infer"
        probe_payload = json.dumps({
            "inputs": [{"name": "PROMPT", "shape": [1, 1], "datatype": "BYTES", "data": ["hi"]}]
        }).encode()

        start = time.time()
        attempt = 0
        while time.time() - start < timeout:
            attempt += 1
            try:
                req = urllib.request.Request(
                    infer_url, data=probe_payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=10) as response:
                    response.read()  # discard result
                logger.info(f"✅ Triton v2 infer endpoint live after {attempt} probe(s).")
                return
            except Exception as e:
                logger.debug(f"   Endpoint probe {attempt}: not ready yet ({e}), waiting 2s...")
                time.sleep(2)
        logger.warning(f"⚠️ Endpoint probe timed out after {timeout}s – proceeding anyway.")

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
                    "shape": [1, 1],
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

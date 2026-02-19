import triton_python_backend_utils as pb_utils
import json
import numpy as np
import os
import shutil
import tempfile

# Impostazione per vLLM (se installato nel container)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _patch_mistral_config(model_name):
    """
    Workaround per bug tra transformers>=4.45 e vLLM<=0.6.x:
    MistralConfig definisce head_dim=None come attributo esplicito.
    vLLM fa getattr(config, "head_dim", fallback) che ritorna None
    invece del fallback, causando TypeError: int * NoneType.

    Soluzione: scarichiamo il config.json, settiamo head_dim=128,
    e restituiamo il path locale patchato.
    """
    from huggingface_hub import snapshot_download

    print(f"[PATCH] Downloading and patching config for {model_name}...")
    local_dir = snapshot_download(
        repo_id=model_name,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*", "*.py"],
    )

    config_path = os.path.join(local_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Calcola head_dim corretto: hidden_size // num_attention_heads
    hidden_size = config.get("hidden_size", 4096)
    num_heads = config.get("num_attention_heads", 32)
    correct_head_dim = hidden_size // num_heads

    if config.get("head_dim") is None:
        config["head_dim"] = correct_head_dim
        print(f"[PATCH] head_dim was None, set to {correct_head_dim}")

    # Forza sliding_window a non-None se necessario (evita altri crash)
    # Mistral v0.3 non usa sliding window, ma il campo null puo' creare problemi
    # Lo lasciamo null perche' vLLM lo gestisce correttamente

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[PATCH] Config patchato salvato in {config_path}")
    return local_dir


class TritonPythonModel:
    def initialize(self, args):
        """
        Inizializzato quando il modello viene caricato.
        """
        self.model_config = json.loads(args['model_config'])
        print(f"[INIT] Inizializzazione Mistral con vLLM Backend (Available: {VLLM_AVAILABLE})")

        model_name = os.environ.get("TRITON_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

        if VLLM_AVAILABLE:
            # Patch config per risolvere head_dim=None bug
            local_model_path = _patch_mistral_config(model_name)

            self.llm = LLM(
                model=local_model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.45,
                max_model_len=2048,  # Ridotto per KV cache (4096 richiede troppa VRAM)
                quantization="gptq",
                dtype="float16",
            )
            self.sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
        else:
            print("[WARN] vLLM non trovato nel container Triton. Caricare Dockerfile.triton!")

    def execute(self, requests):
        """
        Eseguito per ogni richiesta di inferenza.
        """
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = input_tensor.as_numpy().flat[0].decode("utf-8")

            if VLLM_AVAILABLE:
                outputs = self.llm.generate([prompt], self.sampling_params)
                output_text = outputs[0].outputs[0].text
            else:
                output_text = f"[TRITON MOCK MISTRAL] Ricevuto: {prompt}"

            output_tensor = pb_utils.Tensor("RESPONSE", np.array([output_text.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        print("[CLEANUP] Pulizia Mistral backend in corso...")
        if VLLM_AVAILABLE and hasattr(self, 'llm'):
            try:
                import torch
                import gc
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
                
                # Prova a distruggere lo stato parallelo e liberare memoria
                destroy_model_parallel()
                del self.llm
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("[CLEANUP] VRAM liberata con successo per Mistral.")
            except Exception as e:
                print(f"[CLEANUP] Errore durante pulizia: {e}")


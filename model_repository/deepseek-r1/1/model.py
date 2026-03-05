# Questo è il codice che gira dentro Triton Server. È il ponte tra il framework inferenziale ad altissime prestazioni (vLLM) e le API standard di Triton.

# A cosa serve: Inizializza il modello nella VRAM usando vLLM, definisce come i prompt in ingresso devono essere processati e restituisce pezzettini di testo (streaming) generati dalla GPU.
# Perché esiste: Senza questo file, Triton non saprebbe come far girare i modelli open-source complessi che ho scaricato.

import triton_python_backend_utils as pb_utils
import json
import numpy as np
import os

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _patch_deepseek_config(model_name):
    """
    Workaround: TheBloke/deepseek-coder-6.7B-instruct-GPTQ ha
    torch_dtype=bfloat16 nel config.json, ma i kernel GPTQ di vLLM
    supportano solo float16. vLLM legge torch_dtype dal config PRIMA
    di applicare il parametro dtype dell'utente, quindi va patchato.
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

    # Fix: GPTQ richiede float16, non bfloat16
    if config.get("torch_dtype") == "bfloat16":
        config["torch_dtype"] = "float16"
        print("[PATCH] torch_dtype changed from bfloat16 to float16 (GPTQ requirement)")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[PATCH] Config patchato salvato in {config_path}")
    return local_dir


class TritonPythonModel:
    def initialize(self, args):
        # Triton chiama questo metodo quando riceve la richiesta API di caricare il modello. DENTRO questo metodo, ho dovuto istanziare vLLM (self.llm = LLM(...)), che fisicamente va a leggere i pesi dal disco e occupare la memoria GPU.

        self.model_config = json.loads(args['model_config'])
        print(f"[INIT] Inizializzazione DeepSeek-R1 (Ready: {VLLM_AVAILABLE})")

        model_name = os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-ai/deepseek-coder-6.7b-instruct")

        if VLLM_AVAILABLE:
            # Patch config per risolvere bfloat16 + GPTQ incompatibility
            local_model_path = _patch_deepseek_config(model_name)

            self.llm = LLM(
                model=local_model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.45,
                max_model_len=2048,
                quantization="gptq",
                dtype="float16",
                enforce_eager=True,  # Skip CUDA graph capture to avoid race with Triton HTTP routes
            )
            self.sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=512,
                stop=[
                    "```",          # Markdown code fence
                    '"""',          # Python triple-quote docstring
                    "\n\nfrom ",    # any import after a blank line
                    "\ndef ",       # any function definition
                    "\nA: ",        # StackOverflow "Answer:" prefix
                    "\n\n# ",       # any block comment after blank line
                    "# tests",      # test file header
                ]
            )
        else:
            print("[WARN] vLLM non trovato.")
            self.llm = None

    def execute(self, requests):
        # Qui scrivo la logica di generazione testi quando l'utente fa una richiesta chat.
        # Quando riceve una richiesta, estrae il prompt, lo passa a vLLM, e restituisce la risposta.
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = input_tensor.as_numpy().flat[0].decode("utf-8")

            if VLLM_AVAILABLE:
                outputs = self.llm.generate([prompt], self.sampling_params)
                output_text = outputs[0].outputs[0].text
            else:
                output_text = f"[TRITON MOCK DEEPSEEK] {prompt}"

            output_tensor = pb_utils.Tensor("RESPONSE", np.array([output_text.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        # Triton chiama questo metodo quando riceve la richiesta API di scaricare (unload) il modello! Se non scrivo la logica qui, la RAM/VRAM non si libera mai. 

        print("[CLEANUP] Pulizia DeepSeek-R1 backend in corso...")

        # 1. Distruggi l'oggetto LLM
        if VLLM_AVAILABLE and hasattr(self, 'llm'):
            try:
                import torch
                import gc
                # vllm.model_executor.parallel_utils was removed in vLLM >= 0.4.x
                try:
                    from vllm.distributed.parallel_state import destroy_model_parallel
                    destroy_model_parallel()
                except ImportError:
                    pass  # Not needed on single-GPU setups
                del self.llm
                gc.collect() # 2. Forza il garbage collector Python
                torch.cuda.empty_cache() # 3. Chiedo a PyTorch di svuotare la VRAM occupata e ora senza reference
                torch.cuda.synchronize() # 4. Sincronizza la GPU
                print("[CLEANUP] VRAM liberata con successo per DeepSeek-R1.")
            except Exception as e:
                print(f"[CLEANUP] Errore durante pulizia: {e}")


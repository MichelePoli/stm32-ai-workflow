import triton_python_backend_utils as pb_utils
import json
import numpy as np
import os

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        print(f"[INIT] Inizializzazione GPT-OSS 20B (Ready: {VLLM_AVAILABLE})")
        
        # Carica il modello pesante
        model_name = os.environ.get("GPTOSS_MODEL_NAME", "bigcode/starcoder2-15b") # Esempio per 20B (StarCoder2)
        
        if VLLM_AVAILABLE:
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                # Usa 0.6 per stare in ~9.6GB. 
                # Attenzione: richiede lo sload di Mistral/DeepSeek (35%+35%+60% > 100%)
                gpu_memory_utilization=0.6,
                max_model_len=2048,
                quantization="gptq", # Fondamentale per farlo stare in 16GB
                dtype="float16" # Obbligatorio per GPTQ
            )
            self.sampling_params = SamplingParams(temperature=0.2, max_tokens=2048)
        else:
            print("[WARN] vLLM non trovato.")
            self.llm = None

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")
            
            if VLLM_AVAILABLE:
                outputs = self.llm.generate([prompt], self.sampling_params)
                output_text = outputs[0].outputs[0].text
            else:
                output_text = f"[TRITON MOCK GPT-OSS] {prompt}"
            
            output_tensor = pb_utils.Tensor("RESPONSE", np.array([output_text.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        print("[CLEANUP] Pulizia GPT-OSS 20B backend in corso...")

import triton_python_backend_utils as pb_utils
import json
import numpy as np
import os

# Impostazione per vLLM (se installato nel container)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class TritonPythonModel:
    def initialize(self, args):
        """
        Inizializzato quando il modello viene caricato.
        """
        self.model_config = json.loads(args['model_config'])
        print(f"🚀 Inizializzazione Mistral con vLLM Backend (Available: {VLLM_AVAILABLE})")
        
        # Percorso del modello (caricato da environment variable nel docker-compose)
        model_name = os.environ.get("TRITON_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
        
        if VLLM_AVAILABLE:
            # Inizializza l'engine vLLM con limite memoria per lasciare spazio a Ollama
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.7, # Fisso a 0.7 per lasciare ~5GB a Ollama (RTX A4000)
                max_model_len=4096
            )
            self.sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
        else:
            print("⚠️ vLLM non trovato nel container Triton. Caricare Dockerfile.triton!")

    def execute(self, requests):
        """
        Eseguito per ogni richiesta di inferenza.
        """
        responses = []
        for request in requests:
            # Estrae l'input PROMPT
            input_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")
            
            if VLLM_AVAILABLE:
                # Esegue l'inferenza con vLLM
                outputs = self.llm.generate([prompt], self.sampling_params)
                output_text = outputs[0].outputs[0].text
            else:
                # Mock result se vLLM non c'è
                output_text = f"[TRITON MOCK MISTRAL] Ricevuto: {prompt}"
            
            # Prepara il tensor di output
            output_tensor = pb_utils.Tensor("RESPONSE", np.array([output_text.encode("utf-8")], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
            
        return responses

    def finalize(self):
        print("👋 Pulizia Mistral backend in corso...")

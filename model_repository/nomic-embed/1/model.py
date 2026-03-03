import triton_python_backend_utils as pb_utils
import json
import numpy as np
import os

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        print(f"[INIT] Inizializzazione Nomic-Embed (Ready: {SENTENCE_TRANSFORMERS_AVAILABLE})")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            model_name = os.environ.get("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        else:
            print("[WARN] sentence_transformers non trovato.")
            self.model = None

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw = input_tensor.as_numpy()
            # raw shape can be [1] or [1,1] depending on max_batch_size config.
            # The leaf element is bytes (dtype=object) when KIND_CPU backend is used.
            elem = raw.flat[0]
            if isinstance(elem, (bytes, bytearray)):
                text = elem.decode("utf-8")
            elif hasattr(elem, 'item'):
                # numpy scalar wrapping bytes
                text = elem.item().decode("utf-8") if isinstance(elem.item(), bytes) else str(elem.item())
            else:
                text = str(elem)
            
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding = self.model.encode(text)
            else:
                # Mock embedding (768 zeros)
                embedding = np.zeros(768, dtype=np.float32)
            
            output_tensor = pb_utils.Tensor("EMBEDDING", embedding.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses

    def finalize(self):
        print("[CLEANUP] Pulizia Nomic-Embed backend in corso...")

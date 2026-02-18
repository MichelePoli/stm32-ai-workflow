# Architettura Triton Inference Server — Dettagli Tecnici

## 1. Architettura Generale

Il sistema è composto da **due livelli**:

1. **Livello Infrastruttura** (Docker) — i container che girano
2. **Livello Applicazione** (Python/LangGraph) — il codice che li usa

---

## 2. Livello Infrastruttura (Docker Compose)

```
┌──────────────────────────────────────────────────────────────┐
│                     docker-compose.yml                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │   Redis     │  │   MinIO     │  │   Ollama (fallback)  │ │
│  │  :6379      │  │  :9000      │  │   :11434             │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          triton-server (Dockerfile.triton)           │   │
│  │          Porta HOST:8001 → Container:8000 (HTTP)     │   │
│  │          Porta HOST:8002 → Container:8001 (gRPC)     │   │
│  │          Porta HOST:8003 → Container:8002 (Metrics)  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          langgraph-app (Dockerfile)                  │   │
│  │          Porta :8000                                 │   │
│  │          depends_on: triton-server (healthy)         │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

> **Nota sulle porte**: Triton internamente usa `8000` per HTTP. Siccome `langgraph-app`
> usa già la porta `8000` sull'host, il mapping è sfasato: l'host vede Triton su `8001`.
> All'**interno** della rete Docker, i container si parlano direttamente su `triton-server:8000`.

---

## 3. Il Container Triton (`Dockerfile.triton`)

```dockerfile
FROM nvcr.io/nvidia/tritonserver:23.10-py3
# ↑ Immagine base NVIDIA: include CUDA 12.2, cuDNN, e il server Triton già compilato

RUN pip3 install vllm==0.2.7 sentence-transformers accelerate
# ↑ Aggiungiamo:
#   - vLLM: motore di inferenza ad alta performance per LLM (usa PagedAttention)
#   - sentence-transformers: per il modello di embedding Nomic
#   - accelerate: libreria HuggingFace per ottimizzare il caricamento su GPU
```

---

## 4. Il Model Repository

Triton funziona come un **file server di modelli**. Legge una cartella strutturata così:

```
model_repository/
├── mistral/
│   ├── config.pbtxt      ← Configurazione del modello (formato Protocol Buffer)
│   └── 1/                ← Versione 1 del modello
│       └── model.py      ← Il codice Python che esegue l'inferenza
├── deepseek-r1/
│   ├── config.pbtxt
│   └── 1/model.py
├── gpt-oss-20b/
│   ├── config.pbtxt
│   └── 1/model.py
└── nomic-embed/
    ├── config.pbtxt
    └── 1/model.py
```

### `config.pbtxt` — Il contratto I/O del modello

```protobuf
name: "mistral"
backend: "python"          # ← Usa il Python backend di Triton
max_batch_size: 1          # ← Quante richieste processa in parallelo

input [{ name: "PROMPT", data_type: TYPE_STRING, dims: [1] }]
output [{ name: "RESPONSE", data_type: TYPE_STRING, dims: [1] }]

instance_group [{ kind: KIND_GPU, count: 1 }]  # ← Usa la GPU
```

Questo file dice a Triton: *"Questo modello accetta una stringa in input chiamata PROMPT
e restituisce una stringa chiamata RESPONSE"*.

### `model.py` — Il motore di inferenza

Ogni `model.py` implementa l'interfaccia `TritonPythonModel` con 3 metodi obbligatori:

```python
class TritonPythonModel:
    def initialize(self, args):
        # Chiamato UNA VOLTA quando Triton carica il modello in VRAM
        self.llm = LLM(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            gpu_memory_utilization=0.7,  # Usa il 70% della VRAM (≈11GB su 16GB)
        )

    def execute(self, requests):
        # Chiamato per OGNI richiesta di inferenza
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            output = self.llm.generate(prompt)
            return [pb_utils.InferenceResponse(...)]

    def finalize(self):
        # Chiamato quando Triton scarica il modello dalla VRAM
        # Fondamentale per il dynamic model swapping!
        print("Cleanup GPU memory...")
```

---

## 5. Gestione VRAM e Multi-Utente

Con 16GB di VRAM (RTX A4000), tutti i modelli non possono stare in memoria contemporaneamente.
Triton gestisce questo con il **dynamic model swapping**:

```
Scenario: Due utenti simultanei
┌─────────────────────────────────────────────────────┐
│  GPU VRAM (16GB)                                    │
│                                                     │
│  Utente A richiede Mistral:                         │
│  ┌──────────────────┐                               │
│  │   Mistral 7B     │  ~5GB (con gpu_util=0.7)     │
│  └──────────────────┘                               │
│                                                     │
│  Utente B richiede GPT-OSS 20B (quantizzato):       │
│  ┌──────────────────────────────────────────┐       │
│  │   GPT-OSS 20B (GPTQ 4-bit)              │  ~12GB│
│  └──────────────────────────────────────────┘       │
│                                                     │
│  5 + 12 = 17GB > 16GB → CONFLICT!                  │
│                                                     │
│  Triton risolve:                                    │
│  1. Accoda la richiesta di B                        │
│  2. Chiama finalize() su Mistral → libera 5GB       │
│  3. Carica GPT-OSS → 12GB                           │
│  4. Esegue la richiesta di B                        │
│  5. Utente A riceve latenza extra, ma nessun crash  │
└─────────────────────────────────────────────────────┘
```

---

## 6. Livello Applicazione (Python)

### `utils.py` — Il Router Centrale

```python
def get_llm(config, model, ...):
    triton_enabled = os.environ.get("USE_TRITON_BACKEND") == "true"
    triton_models = ["mistral", "deepseek", "gpt-oss"]

    if triton_enabled and any(m in model.lower() for m in triton_models):
        # Normalizza il nome: "mistral-7b-instruct" → "mistral"
        triton_model_name = "mistral"
        if "deepseek" in model: triton_model_name = "deepseek-r1"
        elif "gpt-oss" in model: triton_model_name = "gpt-oss-20b"

        return ChatTriton(triton_url="http://triton-server:8000/v1", model=triton_model_name)
    else:
        return ChatOllama(model=model, ...)  # Fallback

def get_embeddings(...):
    if triton_enabled:
        return TritonEmbeddings(triton_url="http://triton-server:8000/v1")
    else:
        return OllamaEmbeddings(...)  # Fallback
```

### `triton_client.py` — Due Client con API Diverse

**Per i modelli LLM** (`ChatTriton`): usa l'**API OpenAI-compatibile** che vLLM espone
automaticamente su `/v1/chat/completions`.

```
LangGraph → ChatTriton._generate() → POST /v1/chat/completions → vLLM in Triton
```

**Per gli embedding** (`TritonEmbeddings`): usa l'**API nativa di Triton** su
`/v2/models/nomic-embed/infer`, perché il Python backend non espone endpoint OpenAI.

```
LangGraph → TritonEmbeddings._infer() → POST /v2/models/nomic-embed/infer → Triton Python Backend
```

Questo è il motivo per cui esistono due classi separate con logiche diverse.

---

## 7. Flusso Completo di una Richiesta

```
Utente
  │
  ▼
LangGraph Node (es. "generate_code")
  │  chiama get_llm(model="gpt-oss")
  ▼
utils.py → get_llm()
  │  triton_enabled=True, "gpt-oss" in triton_models
  │  triton_model_name = "gpt-oss-20b"
  ▼
ChatTriton(triton_url="http://triton-server:8000/v1", model="gpt-oss-20b")
  │
  ▼
POST http://triton-server:8000/v1/chat/completions
  │  body: {"model": "gpt-oss-20b", "messages": [...]}
  ▼
Triton Server
  │  trova il modello "gpt-oss-20b" nel model_repository
  │  se non è in VRAM → dynamic swap
  ▼
model_repository/gpt-oss-20b/1/model.py → execute()
  │  vLLM genera il testo
  ▼
Risposta JSON → ChatTriton → AIMessage → LangGraph
```

---

## 8. Comandi Utili

```bash
# Build e avvio del sistema completo
docker compose build triton-server
docker compose up -d

# Verifica che Triton sia pronto (health check)
curl http://localhost:8001/v2/health/ready

# Lista dei modelli caricati
curl http://localhost:8001/v2/repository/index

# Log in tempo reale di Triton
docker logs -f stm32-ai-triton

# Metriche Prometheus (per monitoraggio)
curl http://localhost:8003/metrics
```

---

> **In sintesi**: Triton è il "sistema operativo" dei modelli AI — gestisce il caricamento,
> lo scheduling, la memoria GPU e l'API. Il codice Python (LangGraph) non sa nulla di GPU
> o VRAM: chiede semplicemente un modello per nome e Triton si occupa del resto.

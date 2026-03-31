# STM32 AI Workflow

An AI-powered assistant for STM32 embedded development, built with **LangGraph** and backed by **NVIDIA Triton Inference Server** for local LLM inference.

The assistant guides developers through the complete embedded AI pipeline — from STM32 firmware generation to neural network analysis, model customization, fine-tuning, and code integration — through a conversational chat interface inside VS Code.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        VS Code Extension                       │
│               (@stm32ai  chat participant)                      │
└──────────────────────┬─────────────────────────────────────────┘
                       │  HTTP/SSE  (port 8000)
┌──────────────────────▼─────────────────────────────────────────┐
│               FastAPI Server  (src/api/server.py)              │
│         Streaming NDJSON • Session management • Redis          │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│            LangGraph Agent  (src/assistant/graph.py)           │
│                                                                │
│  route_request  ──▶  firmware_flow                             │
│                  ──▶  ai_flow  ──▶  customization ──▶  NNI     │
│                  ──▶  integration_flow                         │
│                  ──▶  search_flow                              │
│                  ──▶  chat                                     │
└──────────────────────┬─────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼────────┐
│    Triton    │ │   Redis    │ │   Ollama    │
│  (Mistral,  │ │ checkpoint │ │  (fallback) │
│  DeepSeek,  │ │  + memory  │ │             │
│  nomic-emb) │ └────────────┘ └─────────────┘
└─────────────┘
```

### Services (Docker Compose)

| Service | Container | Port | Role |
|---|---|---|---|
| Triton Inference Server | `stm32-ai-triton` | 8000 (HTTP), 8001 (gRPC) | Local LLM inference + embeddings |
| Ollama | `stm32-ai-ollama` | 11434 | Fallback LLM backend |
| Redis | `stm32-ai-redis` | 6379 | LangGraph checkpointing + conversation memory |
| LangGraph App | `stm32-ai-langgraph` | 8000 | Python agent + FastAPI server |

---

## Workflows

The agent routes each user request to one of the following workflows:

### 1. Firmware Generation (`firmware_flow`)
Generates a complete STM32 firmware project using STM32CubeMX:
- Installs the required BSP package from the ST repository
- Generates an `.ioc` configuration file via CubeMX script
- Runs STM32CubeMX headlessly to produce the project skeleton

### 2. AI Model Analysis (`ai_flow`)
Drives the full X-CUBE-AI pipeline via **STEdgeAI** (`stedgeai`):
- Searches for recommended models via ChromaDB RAG + DuckDuckGo
- Downloads models from Keras Applications, Hugging Face, or custom paths
- Inspects model architecture (layer count, parameters, FLOPS)
- Runs `analyze`, `validate`, and `generate` commands
- Checks resource constraints (RAM/Flash) before deployment

### 3. Model Customization & Fine-Tuning
After model selection, the agent can:
- **Architecture modification**: add/remove layers, change activation functions, adjust dropout
- **Dataset selection** (Workflow 7): predefined datasets (CIFAR-10, MNIST, HAR…), custom upload, or synthetic
- **Synthetic data generation** (Workflow 6): generate training samples via LLM with validation
- **Fine-tuning**: Keras training with configurable epochs/batch size
- **NNI Hyperparameter Optimization** (Workflow 5): automated search over learning rate, batch size, optimizer using Microsoft NNI
- **VRAM management**: automatically unloads the active Triton LLM before training (freeing ~7.8 GB) and reloads it afterwards

### 4. Code Integration (`integration_flow`)
Merges the generated AI C code into the STM32CubeMX firmware project:
- Scans `st_ai_output/` for generated C files
- Copies and patches `main.c` to include AI initialization and inference calls
- Verifies the integration

### 5. Web Research (`search_flow`)
Searches the web for STM32/AI documentation:
- Direct DuckDuckGo search (no tool-calling required — compatible with Triton backend)
- LLM-based summarization of results
- Academic search via Semantic Scholar / Scholarly

### 6. General Chat
Conversational mode with persistent memory:
- Recalls board, project name, last model across sessions (stored in Redis)
- Supports `reset` / `restart` to wipe all state

---

## Technology Stack

| Component | Technology |
|---|---|
| Agent Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM Backend | NVIDIA Triton Inference Server (primary) + Ollama (fallback) |
| LLM Model | Mistral 7B (vLLM Python backend on Triton) |
| Embedding Model | `nomic-embed-text` via Triton |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| API Layer | FastAPI + SSE streaming |
| Memory / Checkpointing | Redis (AsyncRedisSaver) |
| Hyperparameter Tuning | Microsoft [NNI](https://github.com/microsoft/nni) |
| STM32 Tools | STM32CubeMX (headless) + STEdgeAI (`stedgeai`) |
| VS Code UI | Custom Chat Participant extension (`@stm32ai`) |
| Containerization | Docker Compose |

---

## Prerequisites

### Required Software (on host)

- [Docker](https://www.docker.com/) + Docker Compose
- **STM32CubeMX** installed (provide path via `CUBEMX_PATH`)
- **STEdgeAI** (`stedgeai` binary, provide path via `STEDGEAI_PATH`)
- NVIDIA GPU drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for Triton)
- Conda (optional, for NNI experiments — `stm32` environment)

### Python (for running outside Docker)

- Python 3.11+
- See `requirements.txt` for all dependencies

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/MichelePoli/stm32-ai-workflow.git
cd stm32-ai-workflow
```

### 2. Configure environment

Copy the example environment file and fill in your values:

```bash
cp .env.example .env   # or create .env manually
```

Required variables:

```bash
# STMicroelectronics account (for package download)
ST_EMAIL=your@email.com
ST_PASSWORD=yourpassword

# Triton server URL (set to your GPU server or localhost)
TRITON_BASE_URL=http://localhost:8001

# GitHub token (for model/package search)
GITHUB_ACCESS_TOKEN=ghp_...

# Paths (adapt to your host)
CUBEMX_PATH=/path/to/STM32CubeMX
STEDGEAI_PATH=/path/to/stedgeai
BASE_DIR=/path/to/stm32-ai-workflow/STM32CubeMX
AI_OUTPUT_DIR=/path/to/st_ai_output
```

Optional variables with defaults:

```bash
LOCAL_LLM=mistral                        # LLM model name in Triton
OLLAMA_BASE_URL=http://localhost:11434   # Ollama fallback
USE_TRITON_BACKEND=true
REDIS_URL=redis://localhost:6379
LLM_TEMPERATURE=0
AI_TARGET=stm32f401
AI_COMPRESSION=high
```

> **Note**: The `.env` file is in `.gitignore` and will never be committed.

### 3. Configure Triton model repository

The `model_repository/` directory contains the Triton model configurations.  
Actual model weights must be downloaded separately and placed in the correct subdirectory:

```
model_repository/
├── mistral/          ← Mistral 7B (vLLM Python backend)
│   ├── config.pbtxt
│   └── 1/
│       └── model.py  ← vLLM backend script
├── nomic-embed/      ← nomic-embed-text embedding model
│   ├── config.pbtxt
│   └── 1/
└── deepseek-r1/      ← optional additional LLM
```

Triton runs in **EXPLICIT** control mode — models are loaded/unloaded on demand via API. The `ChatTriton` client in `triton_client.py` handles this automatically, including an OOM fallback that swaps models when VRAM is insufficient.

### 4. Start the services

#### Option A — Full Docker stack (recommended for HPP / GPU server)

```bash
docker compose up -d
```

> ⚠️ The `docker-compose.yml` has volume paths and user UID/GID specific to the original HPP server deployment. **Adapt** `working_dir`, `user`, and volume mounts to your environment before running.

#### Option B — Run FastAPI server on host (dev mode)

Prerequisites: Redis and Triton must already be running.

```bash
./start_server.sh
```

This sets appropriate env vars for local ports and starts the FastAPI server with hot-reload.

### 5. Install the VS Code extension

```bash
cd vscode-extension
npm install
npm run compile
```

Then press **F5** in VS Code to launch the Extension Development Host, or install the `.vsix` package directly:

```bash
code --install-extension vscode-extension/stm32-ai-assistant-0.0.1.vsix
```

Usage inside VS Code Chat: type `@stm32ai <your question>`.

---

## Project Structure

```
stm32-ai-workflow/
├── src/
│   ├── api/
│   │   └── server.py              # FastAPI + SSE streaming endpoint
│   └── assistant/
│       ├── graph.py               # LangGraph master graph + routing
│       ├── state.py               # MasterState dataclass
│       ├── configuration.py       # All settings (from env vars)
│       ├── triton_client.py       # LangChain ↔ Triton bridge (ChatTriton, TritonEmbeddings)
│       ├── utils.py               # get_llm(), ChromaDB, VRAM management helpers
│       ├── workflow1_firmware.py  # STM32CubeMX project generation
│       ├── workflow2_ai.py        # STEdgeAI analysis pipeline
│       ├── workflow3_integration.py # AI code ↔ firmware merge
│       ├── workflow4_web_search.py  # DuckDuckGo search + LLM summarization
│       ├── workflow5_customization.py # Model editing + fine-tuning + NNI
│       ├── workflow6_synthetic_data.py # Synthetic dataset generation
│       ├── workflow7_dataset.py   # Dataset source selection
│       └── nni_optimization/
│           └── generator.py       # NNI trial/manager script generation
├── model_repository/              # Triton model configs (weights excluded)
│   ├── mistral/
│   ├── nomic-embed/
│   └── deepseek-r1/
├── nni_experiments/               # NNI experiment results (scripts)
├── data/                          # Static reference datasets
├── vscode-extension/              # VS Code Chat Participant extension
│   ├── src/extension.ts
│   └── package.json
├── docker-compose.yml             # Full stack definition
├── Dockerfile                     # LangGraph application image
├── langgraph.json                 # LangGraph server config
├── requirements.txt               # Python dependencies
├── pyproject.toml
└── start_server.sh                # Dev server launcher (host-side)
```

---

## VRAM Management

Triton runs in EXPLICIT mode so models are never loaded automatically.  
The `ChatTriton._ensure_model_loaded()` method manages VRAM with a 3-stage strategy:

1. **Fast-path**: if the requested model is already `READY`, do nothing.
2. **Optimistic load**: try to load the model directly.
3. **OOM fallback**: if Triton returns HTTP 400 (VRAM full), unload other LLMs one by one and retry.

Additionally, `workflow5_customization.py` calls `force_unload_triton()` before fine-tuning to unconditionally free VRAM, and `reload_triton_models()` after training completes.

---

## RAG — ChromaDB Knowledge Base

The assistant uses ChromaDB for retrieval-augmented generation when searching for model recommendations:

- **Collections**: `mobilenet`, `yolo`, `har`, `custom`
- **Embeddings**: `nomic-embed-text` via Triton
- **Usage**: `retrieve_best_practices_for_architecture()` queries ChromaDB for architecture-specific fine-tuning tips before presenting modification options to the user

The ChromaDB database is generated at runtime from your documents and is **not included** in this repository. It is automatically rebuilt when the relevant workflow is triggered.

---

## Persistent Memory

User preferences are persisted across sessions via Redis:

- **Short-term** (within session): LangGraph checkpointing in Redis
- **Long-term** (across sessions): `persistent_context` dict stored per `thread_id` in Redis, containing board name, MCU series, last project, last model

A full reset can be triggered by typing `reset` in the chat.

---

## NNI Hyperparameter Optimization

When a user initiates customization, the agent can optionally run **NNI (Neural Network Intelligence)** to automatically search for optimal hyperparameters (learning rate, batch size, optimizer, dropout).

The `nni_optimization/generator.py` module dynamically generates `trial.py` and `manager.py` scripts and launches an NNI experiment. Results are saved under `nni_experiments/`.

---

## Known Limitations

- The `docker-compose.yml` volume paths and user UID are hardcoded for the original HPP server deployment. Adapt them before running on a different machine.
- Triton's Python backend does **not support true streaming** — responses are returned as a single chunk.
- STM32CubeMX must be installed natively on the host; it cannot run inside Docker easily due to GUI/X11 dependencies.
- The `.env` file must be present even in Docker mode (copied into the image at build time by the `Dockerfile`).

---

## License

This project was developed as part of a Master's thesis at **Politecnico di Torino**.  
See the `thesis_mrusso/` directory for documentation.

MIT License — see [LICENSE](LICENSE) if present, otherwise contact the author.

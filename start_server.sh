#!/usr/bin/env bash
# ============================================================
# start_server.sh — Avvia il server FastAPI STM32 AI sull'host
# 
# Usa questo script invece di lanciare il comando manualmente.
# I container Docker (Redis, Triton) devono già girare.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configurazione ---
export USE_TRITON_BACKEND=true
export TRITON_BASE_URL=http://localhost:8001/v1   # Triton espone 8001 sull'host
export REDIS_URL=redis://localhost:6380            # Redis espone 6380 sull'host
export OLLAMA_BASE_URL=http://localhost:11435      # Ollama espone 11435 sull'host
export LOCAL_LLM=mistral

# --- Verifica container attivi ---
echo "🔍 Verifica container Docker..."
for container in stm32-ai-triton stm32-ai-redis; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "❌ Container '$container' non in esecuzione. Avvia i servizi con:"
        echo "   docker compose up -d redis triton-server"
        exit 1
    fi
done
echo "✅ Container ok"

# --- Avvio server ---
echo ""
echo "🚀 Starting STM32 AI Server..."
echo "   Triton : $TRITON_BASE_URL"
echo "   Redis  : $REDIS_URL"
echo "   Ollama : $OLLAMA_BASE_URL"
echo ""

# watchfiles per hot-reload durante sviluppo (come --reload di uvicorn)
python3 -m uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --reload-dir src

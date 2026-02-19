#!/bin/bash
# Precarica i modelli principali in Triton per risposte istantanee
# Lascia GPT-OSS (StarCoder2) scaricato per risparmiare VRAM

BASE_URL="http://localhost:8001/v2/repository/models"

echo "Precaricamento modelli in corso (16GB VRAM Optimized)..."

# Carica i modelli che possono coesistere (~11GB totali)
curl -X POST "${BASE_URL}/nomic-embed/load"
curl -X POST "${BASE_URL}/mistral/load"
curl -X POST "${BASE_URL}/deepseek-r1/load"

echo -e "\n Modelli caricati: Mistral, DeepSeek, Nomic-Embed."
echo "GPT-OSS (StarCoder2) verrà caricato on-demand dall'App quando necessario."

#!/bin/sh
echo "Starting model pull..."

# Main chat model
echo "Pulling mistral..."
ollama pull mistral:latest

# Embeddings model
echo "Pulling nomic-embed-text..."
ollama pull nomic-embed-text:latest

# Evaluation/Reasoning model
echo "Pulling deepseek-r1..."
ollama pull deepseek-r1:latest

# Custom/Large model
echo "Pulling gpt-oss:20b..."
# Note: If this fails, it might be a custom model that needs to be imported or has a different name
ollama pull gpt-oss:20b || echo "Failed to pull gpt-oss:20b (might be custom or unavailable remotely)"

echo "Model pull complete!"

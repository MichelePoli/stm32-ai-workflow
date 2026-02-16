#!/bin/bash

# Configuration
VOLUME_NAME="stm32-ai-workflow_ollama_data"
CONTAINER_NAME="stm32-ai-ollama"
BACKUP_FILE="${1:-ollama_models_backup.tar.gz}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    echo "Usage: ./import_ollama_models.sh <backup_file.tar.gz>"
    exit 1
fi

echo "📦 Starting Ollama Model Import..."

# Check if container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  Stopping Ollama container..."
    docker stop $CONTAINER_NAME
fi

# Ensure volume exists
echo "🛠️  Ensuring volume $VOLUME_NAME exists..."
docker volume create $VOLUME_NAME > /dev/null

# Restore backup
echo "💾 Restoring from $BACKUP_FILE..."
docker run --rm \
  -v $VOLUME_NAME:/data \
  -v $(pwd):/backup \
  alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/$BACKUP_FILE -C /data"

if [ $? -eq 0 ]; then
    echo "✅ Restore completed successfully."
else
    echo "❌ Restore failed!"
    exit 1
fi

echo "🚀 Starting Ollama container..."
docker start $CONTAINER_NAME || echo "⚠️  Container not running (maybe start with docker-compose up -d)"

echo "Done."

#!/bin/bash

# Configuration
VOLUME_NAME="stm32-ai-workflow_ollama_data"
BACKUP_NAME="ollama_models_backup.tar.gz"
CONTAINER_NAME="stm32-ai-ollama"

echo "📦 Starting Ollama Model Export..."

# Check if container is running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "⚠️  Stopping Ollama container to ensure data consistency..."
    docker stop $CONTAINER_NAME
    WAS_RUNNING=true
else
    WAS_RUNNING=false
fi

# Create backup
echo "💾 Creating backup archive ($BACKUP_NAME) from volume $VOLUME_NAME..."
# We use a temporary alpine container to mount the volume and tar it
docker run --rm \
  -v $VOLUME_NAME:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/$BACKUP_NAME -C /data .

if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: $(pwd)/$BACKUP_NAME"
else
    echo "❌ Backup failed!"
    exit 1
fi

# Restart container if it was running
if [ "$WAS_RUNNING" = true ]; then
    echo "🚀 Restarting Ollama container..."
    docker start $CONTAINER_NAME
fi

echo "
🎉 INFO:
To restore this backup on another machine:
1. Ensure docker volume '$VOLUME_NAME' exists (or let docker-compose create it).
2. Run: ./scripts/import_ollama_models.sh $BACKUP_NAME
"

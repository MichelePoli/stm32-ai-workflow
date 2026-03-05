# Questo è il file di build dell'applicazione principale LangGraph (l'assistente AI in Python). Contiene tutta la configurazione per installare TensorFlow, compilare pacchetti e montare i volumi con il tuo utente Linux (mrusso UID 1002).

# ============================================================
# Dockerfile — STM32 AI LangGraph Application Server
# 
# Uses ubuntu:22.04 (same base as host) so that host-mounted
# binaries (Miniconda, stedgeai) run without glibc mismatches.
# User 'mrusso' is created with the same UID (1002) as the host
# user to avoid permission issues on mounted volumes.
# ============================================================




FROM ubuntu:22.04

# Prevent interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python / process settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libgomp1 \
    xvfb \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxext6 \
    libx11-6 \
    libfreetype6 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 and pip point to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create user with same UID as host (1002) so mounted volumes have correct perms
ARG USER_UID=1002
ARG USER_GID=1002
RUN groupadd -g $USER_GID mrusso && \
    useradd -u $USER_UID -g $USER_GID -m -s /bin/bash mrusso

# Working directory — matches host path for absolute path consistency
WORKDIR /home/mrusso/stm32-ai-workflow

# Install Python dependencies as root first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (will be overridden by volume mount in dev mode)
COPY src/ src/
COPY .env .env

# Switch to host-matching user
USER mrusso

# Expose FastAPI port
EXPOSE 8000

# Use watchfiles for hot-reload during development
CMD ["python3", "-m", "uvicorn", "src.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--reload", "--reload-dir", "src"]

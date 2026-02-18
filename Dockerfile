# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY .env .env

# Expose port
EXPOSE 8000

# Command to run the application
# We use uvicorn to serve the FastAPI app
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

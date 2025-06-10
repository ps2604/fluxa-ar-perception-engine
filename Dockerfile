FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUPY_CACHE_DIR=/tmp/cupy_cache

WORKDIR /app

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip build-essential git libgl1-mesa-glx libglib2.0-0 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install the FLUXA package
COPY . .
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /tmp/cupy_cache /app/checkpoints /app/outputs

# Set entrypoint to use the package command
ENTRYPOINT ["fluxa-train"]

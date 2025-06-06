###############################################################################
# FlowField – FSE Native – Multi-GPU Training Image
# CUDA 11.8 / cuDNN 8 / NCCL  •  Ubuntu 22.04  •  Python 3.10
###############################################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUPY_CACHE_DIR=/tmp/cupy_cache
ENV CUPY_TF32=1
ENV CUPY_ACCELERATORS=cub,cutensor
ENV NCCL_DEBUG=INFO
ENV NCCL_SHM_DISABLE=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_SOCKET_IFNAME=^docker0,lo
ENV NCCL_BLOCKING_WAIT=1
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV OMP_NUM_THREADS=1

WORKDIR /app

# System packages - handle NCCL installation carefully
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-client \
    openssh-server \
    net-tools \
    iputils-ping \
    libnuma1 \
    libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install or upgrade NCCL packages (handle held packages)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libnccl2 libnccl-dev || \
    apt-get install -y --no-install-recommends --allow-change-held-packages libnccl2 libnccl-dev) && \
    rm -rf /var/lib/apt/lists/*

# Create python symlink
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -r requirements.txt

# Copy FlowField source code
COPY setup.py .
COPY flowfield_core_optimized.py .
COPY flowfield_components.py .
COPY flowfield_fluxa_model.py .
COPY flowfield_training_ultra_optimized.py .
COPY flowfield_async_data_loader.py .
COPY flowfield_advanced_cuda_kernels.py .
COPY metrics_fse.py .

# Install FlowField package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /tmp/cupy_cache /app/checkpoints /app/outputs && \
    chmod +x /app/flowfield_training_ultra_optimized.py

# Validate installation
RUN echo "=== NCCL Status ===" && \
    dpkg -l | grep nccl || echo "No NCCL packages found via dpkg" && \
    python3 -c "import cupy as cp; print(f'CuPy: {cp.__version__}'); print(f'CUDA: {cp.cuda.runtime.runtimeGetVersion()}')" || echo "CuPy test failed" && \
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch test failed" && \
    python3 -c "from cupy.cuda import nccl; print(f'NCCL: {nccl.get_build_version()}'); nccl.get_unique_id(); print('NCCL test passed')" || echo "NCCL test failed"

# Set entrypoint
ENTRYPOINT ["python3", "/app/flowfield_training.py"]
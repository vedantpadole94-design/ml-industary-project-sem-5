FROM nvidia/cuda:12.1.0-base-ubuntu22.04

LABEL maintainer="MARL Research Team"
LABEL description="Multi-Agent Reinforcement Learning — Cooperative Grid World"

# Build arguments
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev,render]"

# Create runs directory
RUN mkdir -p runs/ logs/ checkpoints/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV RAY_DISABLE_MEMORY_MONITOR=1

# Default seed
ENV MARL_SEED=42

# Expose TensorBoard port
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import marl; print('OK')" || exit 1

# Default command: run full experiment suite
CMD ["bash", "run_all.sh"]

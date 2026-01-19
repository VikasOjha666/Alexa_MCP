# ================================
# GPU-enabled llama.cpp + FastAPI
# ================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# ----------------
# System packages
# ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    cmake \
    pkg-config \
    libopenblas-dev \
    libgomp1 \
    libcurl4-openssl-dev \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# ----------------
# CUDA build flags for llama.cpp
# ----------------
ENV CMAKE_ARGS="-DLLAMA_CUDA=on -DGGML_CUDA_NO_VMM=1"
ENV FORCE_CMAKE=1
ENV LLAMA_CUDA=1
ENV LLAMA_CUDA_FORCE_MMQ=1
ENV LLAMA_CUDA_FORCE_CUBLAS=1

# Python runtime flags
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# ----------------
# Create app user
# ----------------
RUN useradd --create-home appuser
WORKDIR /home/appuser/app

# ----------------
# Install Python deps (as root)
# ----------------
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r requirements.txt
 
RUN pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121





# Sanity check (fail fast if MCP deps are broken)
RUN python -c "from llama_index.tools.mcp import BasicMCPClient, McpToolSpec; print('llama_index.tools.mcp import OK')"

# ----------------
# Copy app code
# ----------------
COPY --chown=appuser:appuser . .

USER appuser

# ----------------
# Runtime config
# ----------------
EXPOSE 3000

ENV MODEL_PATH=/home/appuser/app/gorilla-openfunctions-v2-GGUF/gorilla-openfunctions-v2-q4_K_M.gguf
ENV MCP_SERVER_URLS="http://mcp_calculator:8000/sse,http://mcp_converter:8000/sse"

# ----------------
# Start FastAPI
# ----------------
CMD ["python","-m","uvicorn","llm_client:app","--host","0.0.0.0","--port","3000","--workers","1"]

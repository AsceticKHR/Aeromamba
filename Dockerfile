# ==============================================================================
# AeroMamba-VLA Dockerfile
# Mirrors the AutoDL Environment: Ubuntu 22.04 + CUDA 11.8 + Conda env 'mamba2'
# ==============================================================================

# Use CUDA 11.8 devel image as base (devel is required to build custom CUDA kernels for mamba)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

# Install core system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh

# Configure Conda & add mirrors (Tsinghua mirrors for fast domestic download if needed)
RUN conda init bash && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes

# Create the python environment matching your AutoDL setup (mamba2)
RUN conda create -n mamba2 python=3.10.14 -y

# Install PyTorch 2.1.1 + CUDA 11.8 (strictly matching AutoDL container)
RUN conda run -n mamba2 pip install --no-cache-dir \
    torch==2.1.1+cu118 \
    torchvision==0.16.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install Ninja (highly recommended for parallel compiling Mamba extensions) and packaging
RUN conda run -n mamba2 pip install --no-cache-dir ninja packaging einops

# Compile & Install causal-conv1d and mamba-ssm inside the conda env
# CRITICAL: We use --no-build-isolation so pip uses the pre-installed GPU PyTorch for compiling
RUN conda run -n mamba2 pip install causal-conv1d==1.4.0 --no-build-isolation && \
    conda run -n mamba2 pip install mamba-ssm==2.2.2 --no-build-isolation

# Install remaining core requirements exported from your environment
RUN conda run -n mamba2 pip install --no-cache-dir \
    transformers==4.42.3 \
    peft==0.11.1 \
    timm==0.9.16 \
    Flask==3.1.3 \
    pillow==12.2.0 \
    numpy==1.26.4 \
    safetensors==0.8.0 \
    accelerate==1.14.0 \
    tqdm==4.68.3 \
    requests==2.34.2 \
    pyyaml==6.0.3

# Set the mamba2 env as default by prepending its bin folder to PATH
ENV PATH="/root/miniconda3/envs/mamba2/bin:${PATH}"

# Set workspace
WORKDIR /workspace

# Expose port for AeroMamba flask server (inference)
EXPOSE 5007

# Default command
CMD ["/bin/bash"]

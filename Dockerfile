# Use CUDA 11.3.1 with Ubuntu 20.04
FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for Python
ENV PYTHON_VERSION=3.7

# Install basic system packages, Python, and dependencies in a single RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    apt-utils \
    git \
    curl \
    vim \
    unzip \
    openssh-client \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libffi-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Configure Python environment and upgrade pip
RUN rm -f /usr/bin/python /usr/bin/pip || true && \
    ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip setuptools

# Set up Python aliases for convenience
RUN echo "alias python='python${PYTHON_VERSION}'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

# Install essential Python libraries
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pyyaml \
    cffi \
    matplotlib \
    Cython \
    requests \
    opencv-python \
    "pillow<7"

# Install PyTorch and torchvision compatible with CUDA 11.3
RUN pip install --no-cache-dir \
    torch==1.10.0+cu113 \
    torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory to DenseFusion
WORKDIR /root/dense_fusion

# Expose port for TensorBoard
EXPOSE 6006

# Automatically navigate to the working directory on login
RUN echo "cd /root/dense_fusion" >> /root/.bashrc

# Set default command to bash
CMD ["/bin/bash"]
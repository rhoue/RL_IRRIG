# Dockerfile for RL Intelligent Irrigation Project
# Ubuntu-based image with data science libraries

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    # Python and pip
    python3.10 \
    python3.10-dev \
    python3-pip \
    # Build essentials for compiling Python packages
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # System libraries for data science packages
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    # Graphics and visualization libraries
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libfreetype6-dev \
    libxft-dev \
    # Scientific computing libraries
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    # Additional utilities
    wget \
    curl \
    git \
    vim \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python and pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-prod.txt requirements-dev.txt ./

# Install Python dependencies from requirements files
# Install production dependencies first
RUN pip install --no-cache-dir -r requirements-prod.txt

# Install PyTorch (CPU version - change to CUDA if GPU needed)
# Note: PyTorch is in requirements-dev.txt but we install separately for index URL
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install development dependencies (includes gymnasium, stable-baselines3)
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional RL libraries with extras
RUN pip install --no-cache-dir stable-baselines3[extra] sb3-contrib

# Install Streamlit (if not already in requirements)
RUN pip install --no-cache-dir streamlit

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
# Users can override this to run other scripts
CMD ["streamlit", "run", "src/rl_intelli_irrig_streamlit_config.py", "--server.port=8501", "--server.address=0.0.0.0"]

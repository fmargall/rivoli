# Base image: Ubuntu 24.04 LTS, supported until 2029
FROM ubuntu:24.04

# Prevent apt from asking interactive questions during build
ENV DEBIAN_FRONTEND=noninteractive

# Install C++ build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Required for building C++ projects
    build-essential \
    gcc-13 \
    g++-13 \
    cmake \
    ninja-build \
    git \
    # Python for nanobind binding
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Useful utilies for debugging
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Force GCC 13 as the default compiler
# (in case an older version is around, and to be explicit)
RUN    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Create a Python venv to avoid Ubuntu 24.04'
# "externally-managed-environment" error, and
# install the binding's dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    nanobind \
    scikit-build-core \
    numpy \
    pytest

# Working directory inside the container
WORKDIR /rivoli

# Open a shell by default when running the container interactively
CMD ["/bin/bash"]

# UnifyWeaver Development Environment
# Ubuntu-based container with SWI-Prolog, Node.js, Claude Code, and development tools

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install system dependencies and common utilities
RUN apt-get update && apt-get install -y \
    # Core utilities
    curl \
    wget \
    git \
    vim \
    nano \
    unzip \
    ca-certificates \
    gnupg \
    lsb-release \
    # Build tools
    build-essential \
    cmake \
    # Text processing utilities
    grep \
    sed \
    gawk \
    ripgrep \
    jq \
    xmlstarlet \
    libxml2-utils \
    # Shell
    bash \
    bash-completion \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    python3-lxml \
    python3-numpy \
    # Databases
    sqlite3 \
    # Go
    golang \
    # SWI-Prolog dependencies
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install onnxruntime

# Install SWI-Prolog
RUN apt-add-repository ppa:swi-prolog/stable && \
    apt-get update && \
    apt-get install -y swi-prolog && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PowerShell
RUN apt-get update && apt-get install -y wget apt-transport-https software-properties-common && \
    wget -q "https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb" && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y powershell && \
    apt-get install -y dotnet-sdk-8.0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI and Gemini CLI globally
RUN npm install -g @anthropic-ai/claude-code @google/gemini-cli

# Create working directory matching WSL path structure
RUN mkdir -p /mnt/c/Users/johnc/Dropbox/projects

# Set working directory
WORKDIR /mnt/c/Users/johnc/Dropbox/projects/UnifyWeaver

# Set up bash as default shell with better prompt
RUN echo 'PS1="\[\033[01;32m\]\u@unifyweaver-dev\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc

# Install useful bash aliases
RUN echo 'alias ll="ls -lah"' >> /root/.bashrc && \
    echo 'alias swipl="swipl -q"' >> /root/.bashrc && \
    echo 'alias test-stream="swipl -g \"use_module(src/unifyweaver/core/stream_compiler), test_stream_compiler, halt.\""' >> /root/.bashrc && \
    echo 'alias test-recursive="swipl -g \"use_module(src/unifyweaver/core/recursive_compiler), test_recursive_compiler, halt.\""' >> /root/.bashrc && \
    echo 'alias test-advanced="swipl -g \"use_module(src/unifyweaver/core/advanced/test_advanced), test_all_advanced, halt.\""' >> /root/.bashrc

# Set up git configuration to match host
RUN git config --global --add safe.directory '*'

# Expose any ports that might be needed (adjust as necessary)
EXPOSE 3000 8080

# Default command - start bash
CMD ["/bin/bash"]

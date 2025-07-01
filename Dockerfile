FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /workspace/aiams

# Install system dependencies with flags to prevent interactive prompts
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    libreoffice \
    libreoffice-java-common \
    default-jre \
    tzdata \
    gosu \
    && echo "${TZ}" > /etc/timezone \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /workspace/aiams/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Set environment variables
ENV PYTHONPATH=/workspace/aiams
ENV PYTHONUNBUFFERED=1


# Set the entrypoint and default command
CMD ["bash"]

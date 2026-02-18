FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WANDB_DISABLED=true

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip

# Install CUDA-compatible PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install rest of dependencies (WITHOUT torch inside requirements.txt)
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/

RUN mkdir -p data outputs results logs

CMD ["python3", "src/train.py"]

# Mini-LLM training image — Linux + Python + CUDA PyTorch (NVIDIA GPU via Docker)
# CPU-only: docker build --build-arg TORCH_INDEX=cpu -t mini-ml:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# TORCH_INDEX=cu124 → CUDA 12.x wheels (needs NVIDIA driver + GPU passed into container)
# TORCH_INDEX=cpu   → CPU-only wheel (smaller; no GPU in container)
ARG TORCH_INDEX=cu124
RUN pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
 && pip install sentencepiece

COPY . .

# Default: run training (override with docker run ... python scripts/generate.py ...)
CMD ["python", "scripts/train.py"]

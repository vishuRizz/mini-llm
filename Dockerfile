# Mini-LLM training image — Linux + Python + CPU PyTorch (works on Windows/macOS/Linux via Docker)
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch + SentencePiece (matches requirements.txt without conflicting torch sources)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install sentencepiece

COPY . .

# Default: run training (override with docker run ... python scripts/generate.py ...)
CMD ["python", "scripts/train.py"]

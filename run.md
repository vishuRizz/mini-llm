cd ~/Desktop/mini-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train:
# - auto-cleans data/train.csv + data/validation.csv (or data/val.csv) + data/test.csv
# - writes processed files to data/processed/
# - trains SentencePiece tokenizer to artifacts/tokenizer.model
# - trains model and saves best checkpoint to artifacts/tiny_llm.pt
python3 scripts/train.py

# Generate one response
python3 scripts/generate.py --prompt "hi" --tokens 80 --temperature 0.8 --top-k 40

# Interactive chat
python scripts/chat.py

# Docker (Windows-friendly): see DOCKER.md
# docker compose build && docker compose run --rm train







Build the image
cd mini-ml
docker build -t mini-ml:latest .
Or:

docker compose build
Train (recommended: Compose so outputs stay on your repo)
docker compose run --rm train
This mounts the current directory to /app, so artifacts/, data/processed/, etc. are written on your machine.

Windows CMD:

cd mini-ml
docker compose run --rm train
Windows PowerShell (if you use docker run manually):

docker run --rm -v "${PWD}:/app" mini-ml:latest
After training: generate
docker compose --profile tools run --rm generate
Or:

docker run --rm -v "${PWD}:/app" mini-ml:latest python scripts/generate.py --prompt "hi" --tokens 80

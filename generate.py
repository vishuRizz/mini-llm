import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from mini_transformer import build_tiny_decoder_only_transformer


VOCAB_FILE = "vocab.json"
MODEL_FILE = "tiny_llm.pt"
BLOCK_SIZE = 64
DEFAULT_TEMPERATURE = 0.8
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


def load_vocab(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in data["stoi"].items()}
    # itos keys were stringified when saved
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    ids = [stoi[ch] for ch in text if ch in stoi]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: torch.Tensor, itos: Dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)


@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int, temperature: float = DEFAULT_TEMPERATURE) -> str:
    if not Path(VOCAB_FILE).exists() or not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            "Model or vocab not found. Run train_char_lm.py first to create tiny_llm.pt and vocab.json."
        )

    stoi, itos = load_vocab(VOCAB_FILE)
    vocab_size = len(stoi)

    model = build_tiny_decoder_only_transformer(vocab_size=vocab_size, max_len=BLOCK_SIZE)
    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    x = encode(prompt, stoi)
    if x.numel() == 0:
        raise ValueError("Prompt contains no known characters from the training text.")

    x = x.unsqueeze(0).to(DEVICE)  # (1, seq_len)

    for _ in range(max_new_tokens):
        seq_len = x.size(1)
        if seq_len > BLOCK_SIZE:
            x_cond = x[:, -BLOCK_SIZE:]
        else:
            x_cond = x

        logits = model(x_cond)  # (1, seq_len, vocab_size)
        last_logits = logits[:, -1, :]  # (1, vocab_size)
        last_logits = last_logits / temperature
        probs = torch.softmax(last_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        x = torch.cat([x, next_id], dim=1)

    return decode(x[0].cpu(), itos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with tiny decoder-only Transformer")
    parser.add_argument("--prompt", type=str, default="The ", help="Prompt string to start generation")
    parser.add_argument("--tokens", type=int, default=100, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (default 0.8)")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Prompt: {repr(args.prompt)}, temperature: {args.temperature}")
    text = generate_text(args.prompt, args.tokens, temperature=args.temperature)
    print("\n=== Generated Text ===\n")
    print(text)


if __name__ == "__main__":
    main()


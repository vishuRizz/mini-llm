import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mini_transformer import build_tiny_decoder_only_transformer


DATA_FILE = "tiny-shakespeare.txt"
VOCAB_FILE = "vocab.json"
MODEL_FILE = "tiny_llm.pt"

BLOCK_SIZE = 64
BATCH_SIZE = 128
NUM_STEPS = 20000
LEARNING_RATE = 3e-4
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Please create it and add some text for training."
        )
    return p.read_text(encoding="utf-8")


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


class CharDataset(Dataset):
    def __init__(self, ids: List[int], block_size: int) -> None:
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def main() -> None:
    print(f"Using device: {DEVICE}")
    raw_text = read_text(DATA_FILE)
    print(f"Loaded {len(raw_text)} characters from {DATA_FILE}")

    stoi, itos = build_vocab(raw_text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    encoded = encode(raw_text, stoi)
    dataset = CharDataset(encoded, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = build_tiny_decoder_only_transformer(vocab_size=vocab_size, max_len=BLOCK_SIZE)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    step = 0
    model.train()
    while step < NUM_STEPS:
        for xb, yb in dataloader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 1000 == 0 or step == 1:
                print(f"Step {step}/{NUM_STEPS}, loss = {loss.item():.4f}")

            if step >= NUM_STEPS:
                break

    torch.save(model.state_dict(), MODEL_FILE)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f, ensure_ascii=False)

    print(f"Training complete. Model saved to {MODEL_FILE}, vocab to {VOCAB_FILE}")


if __name__ == "__main__":
    main()


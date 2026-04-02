import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_pipeline import process_all_splits
from .model import build_tiny_decoder_only_transformer
from .runtime import pick_device
from .tokenizer_utils import encode_text, load_tokenizer, train_sentencepiece


ARTIFACTS_DIR = Path("artifacts")
MODEL_FILE = ARTIFACTS_DIR / "tiny_llm.pt"
TOKENIZER_MODEL = ARTIFACTS_DIR / "tokenizer.model"
TRAINED_CONFIG = ARTIFACTS_DIR / "training_config.json"

BLOCK_SIZE = 128
BATCH_SIZE = 32
NUM_STEPS = 12000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
VOCAB_SIZE = 3000
WARMUP_STEPS = 800
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.05
EVAL_EVERY = 300
EVAL_BATCHES = 20
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
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


def lr_for_step(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * float(step + 1) / float(max(1, WARMUP_STEPS))
    progress = (step - WARMUP_STEPS) / float(max(1, NUM_STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    min_lr = LEARNING_RATE * 0.1
    return min_lr + (LEARNING_RATE - min_lr) * cosine


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: Dataset,
    criterion: nn.Module,
    device: str,
    batch_size: int,
    eval_batches: int,
) -> float:
    if len(dataset) == 0:
        return float("inf")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    model.eval()
    losses: List[float] = []
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        losses.append(float(loss.item()))
        if i + 1 >= eval_batches:
            break
    model.train()
    if not losses:
        return float("inf")
    return sum(losses) / len(losses)


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("Processing train/val/test CSV files...")
    split_files = process_all_splits()
    train_text_path = split_files["train"]
    val_text_path = split_files["val"]
    test_text_path = split_files["test"]

    print(f"Training tokenizer on: {train_text_path}")
    train_sentencepiece(
        input_text_path=train_text_path,
        model_prefix=ARTIFACTS_DIR / "tokenizer",
        vocab_size=VOCAB_SIZE,
    )
    sp = load_tokenizer(TOKENIZER_MODEL)
    vocab_size = int(sp.get_piece_size())
    print(f"Tokenizer vocab size: {vocab_size}")

    train_ids = encode_text(sp, train_text_path.read_text(encoding="utf-8"))
    val_ids = encode_text(sp, val_text_path.read_text(encoding="utf-8"))
    test_ids = encode_text(sp, test_text_path.read_text(encoding="utf-8"))

    train_dataset = TokenDataset(train_ids, BLOCK_SIZE)
    val_dataset = TokenDataset(val_ids, BLOCK_SIZE)
    test_dataset = TokenDataset(test_ids, BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = build_tiny_decoder_only_transformer(vocab_size=vocab_size, max_len=BLOCK_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    step = 0
    best_val_loss = float("inf")
    while step < NUM_STEPS:
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            lr = lr_for_step(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            step += 1
            if step % EVAL_EVERY == 0 or step == 1:
                val_loss = estimate_loss(
                    model=model,
                    dataset=val_dataset,
                    criterion=criterion,
                    device=device,
                    batch_size=BATCH_SIZE,
                    eval_batches=EVAL_BATCHES,
                )
                print(
                    f"Step {step}/{NUM_STEPS}, lr={lr:.6f}, train_loss={loss.item():.4f}, val_loss={val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), MODEL_FILE)
                    print(f"Saved new best checkpoint to {MODEL_FILE}")
            if step >= NUM_STEPS:
                break

    if not MODEL_FILE.exists():
        torch.save(model.state_dict(), MODEL_FILE)

    test_loss = estimate_loss(
        model=model,
        dataset=test_dataset,
        criterion=criterion,
        device=device,
        batch_size=BATCH_SIZE,
        eval_batches=EVAL_BATCHES,
    )

    with TRAINED_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "block_size": BLOCK_SIZE,
                "batch_size": BATCH_SIZE,
                "num_steps": NUM_STEPS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS,
                "max_grad_norm": MAX_GRAD_NORM,
                "label_smoothing": LABEL_SMOOTHING,
                "vocab_size": vocab_size,
                "tokenizer_model": str(TOKENIZER_MODEL),
                "best_val_loss": best_val_loss,
                "test_loss": test_loss,
                "device": device,
            },
            f,
            indent=2,
        )
    print(f"Training complete. best_val_loss={best_val_loss:.4f}, test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()

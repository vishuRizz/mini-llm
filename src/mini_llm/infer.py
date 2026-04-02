from pathlib import Path

import torch

from .model import build_tiny_decoder_only_transformer
from .runtime import pick_device
from .tokenizer_utils import decode_ids, encode_text, load_tokenizer


ARTIFACTS_DIR = Path("artifacts")
TOKENIZER_MODEL = ARTIFACTS_DIR / "tokenizer.model"
MODEL_FILE = ARTIFACTS_DIR / "tiny_llm.pt"
BLOCK_SIZE = 128


def load_model_and_tokenizer() -> tuple[torch.nn.Module, object, str]:
    if not TOKENIZER_MODEL.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Model or tokenizer not found. Run training first.")
    device = pick_device()
    sp = load_tokenizer(TOKENIZER_MODEL)
    vocab_size = int(sp.get_piece_size())
    model = build_tiny_decoder_only_transformer(vocab_size=vocab_size, max_len=BLOCK_SIZE)
    state_dict = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, sp, device


@torch.no_grad()
def generate_reply(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    model, sp, device = load_model_and_tokenizer()
    wrapped_prompt = f"<|user|> {prompt}\n<|assistant|>"
    token_ids = encode_text(sp, wrapped_prompt)
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -BLOCK_SIZE:] if x.size(1) > BLOCK_SIZE else x
        logits = model(x_cond)[:, -1, :]
        logits = logits / max(temperature, 1e-5)
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits[logits < values[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    decoded = decode_ids(sp, x[0].cpu().tolist())
    if "<|assistant|>" in decoded:
        decoded = decoded.split("<|assistant|>", maxsplit=1)[1]
    if "<|user|>" in decoded:
        decoded = decoded.split("<|user|>", maxsplit=1)[0]
    return decoded.strip()

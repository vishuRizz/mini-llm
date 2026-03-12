import json
from pathlib import Path

import torch

from mini_transformer import build_tiny_decoder_only_transformer


VOCAB_FILE = "vocab.json"
MODEL_FILE = "tiny_llm.pt"
BLOCK_SIZE = 64
TEMPERATURE = 0.8
DEVICE = "cpu"


def load_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi = {k: int(v) if isinstance(v, str) and v.isdigit() else v for k, v in data["stoi"].items()}
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def encode(text: str, stoi):
    ids = [stoi[ch] for ch in text if ch in stoi]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


def main() -> None:
    if not Path(VOCAB_FILE).exists() or not Path(MODEL_FILE).exists():
        raise FileNotFoundError(
            "Model or vocab not found. Run train_char_lm.py first to create tiny_llm.pt and vocab.json."
        )

    print(f"Loading model from {MODEL_FILE} on {DEVICE}...")
    stoi, itos = load_vocab(VOCAB_FILE)
    vocab_size = len(stoi)

    model = build_tiny_decoder_only_transformer(vocab_size=vocab_size, max_len=BLOCK_SIZE)
    state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("Ready. Type your message and press Enter. Type 'quit' to exit.\n")

    history = ""
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        # Simple conversation format in the prompt
        history += f"You: {user_input}\nModel:"

        # Encode history (truncate to last BLOCK_SIZE chars to keep context small)
        context = history[-BLOCK_SIZE:]
        x = encode(context, stoi)
        if x.numel() == 0:
            print("Model: (cannot understand input characters based on training text)")
            continue

        x = x.unsqueeze(0).to(DEVICE)

        # Generate a short reply
        max_new_tokens = 200
        with torch.no_grad():
            for _ in range(max_new_tokens):
                seq_len = x.size(1)
                if seq_len > BLOCK_SIZE:
                    x_cond = x[:, -BLOCK_SIZE:]
                else:
                    x_cond = x

                logits = model(x_cond)
                last_logits = logits[:, -1, :]
                last_logits = last_logits / TEMPERATURE
                probs = torch.softmax(last_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_id], dim=1)

        generated = decode(x[0].cpu(), itos)

        # Extract only the part after the last "Model:" to print as reply
        if "Model:" in generated:
            reply = generated.split("Model:", maxsplit=1)[1]
        else:
            reply = generated

        # Cut reply at the first newline for a shorter answer
        reply_line = reply.split("\n", maxsplit=1)[0].strip()
        print(f"Model: {reply_line}\n")

        history += reply + "\n"


if __name__ == "__main__":
    main()


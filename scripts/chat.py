import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mini_llm.infer import generate_reply
from src.mini_llm.runtime import pick_device


def main() -> None:
    print(f"Using device: {pick_device()}")
    print("Ready. Type your message and press Enter. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        reply = generate_reply(prompt=user_input, max_new_tokens=80, temperature=0.8, top_k=40)
        print(f"Model: {reply}\n")


if __name__ == "__main__":
    main()

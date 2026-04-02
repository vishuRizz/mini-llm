from pathlib import Path
from typing import List

import sentencepiece as spm


def train_sentencepiece(
    input_text_path: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float = 1.0,
) -> Path:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_text_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
        user_defined_symbols=["<|user|>", "<|assistant|>"],
    )
    return model_prefix.with_suffix(".model")


def load_tokenizer(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode_text(sp: spm.SentencePieceProcessor, text: str) -> List[int]:
    return list(sp.encode(text, out_type=int))


def decode_ids(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    return sp.decode(ids)

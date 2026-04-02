import ast
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List


RAW_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
SPLIT_CANDIDATES: Dict[str, List[str]] = {
    "train": ["train.csv"],
    "val": ["val.csv", "validation.csv"],
    "test": ["test.csv"],
}


def resolve_split_path(split: str, data_dir: Path = RAW_DATA_DIR) -> Path:
    for candidate in SPLIT_CANDIDATES[split]:
        path = data_dir / candidate
        if path.exists():
            return path
    expected = ", ".join(SPLIT_CANDIDATES[split])
    raise FileNotFoundError(f"Could not find {split} split. Expected one of: {expected}")


def _fallback_extract(dialog_text: str) -> List[str]:
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", dialog_text)
    return [a or b for a, b in quoted if (a or b).strip()]


def parse_dialog(dialog_text: str) -> List[str]:
    try:
        parsed = ast.literal_eval(dialog_text)
        if isinstance(parsed, list):
            turns = [str(x).strip() for x in parsed if str(x).strip()]
            if len(turns) >= 2:
                return turns
    except (ValueError, SyntaxError):
        pass

    turns: List[str] = []
    for raw_line in dialog_text.splitlines():
        line = raw_line.strip().strip("[]").strip().rstrip(",")
        if len(line) < 2:
            continue
        if line[0] in {"'", '"'}:
            line = line[1:]
            if line.endswith(("'", '"')):
                line = line[:-1]
        line = line.strip()
        if line:
            turns.append(line)
    if len(turns) >= 2:
        return turns
    return _fallback_extract(dialog_text)


def clean_text(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u3002": ".",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.replace("''", "'").replace('""', '"')
    text = text.strip(" '\"")
    text = re.sub(r"\s['\"]\s", " ", text)
    text = re.sub(r"(^|\s)['\"](?=\s|$)", " ", text)
    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def read_dialogs(csv_path: Path) -> Iterable[List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "dialog" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain a 'dialog' column.")
        for row in reader:
            dialog_text = row.get("dialog", "")
            turns = [clean_text(t) for t in parse_dialog(dialog_text)]
            turns = [t for t in turns if t]
            if len(turns) >= 2:
                yield turns


def build_examples(turns: List[str], context_turns: int = 6) -> List[str]:
    examples: List[str] = []
    for i in range(1, len(turns), 2):
        start = max(0, i - context_turns)
        context = turns[start:i]
        reply = turns[i]
        lines: List[str] = []
        for j, utterance in enumerate(context):
            speaker = "<|user|>" if (start + j) % 2 == 0 else "<|assistant|>"
            lines.append(f"{speaker} {utterance}")
        lines.append(f"<|assistant|> {reply}")
        examples.append("\n".join(lines))
    return examples


def process_split(split: str, data_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    source_path = resolve_split_path(split, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.txt"
    examples: List[str] = []
    for turns in read_dialogs(source_path):
        examples.extend(build_examples(turns))
    unique_examples = list(dict.fromkeys(examples))
    out_path.write_text("\n\n".join(unique_examples), encoding="utf-8")
    return out_path


def process_all_splits(data_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR) -> Dict[str, Path]:
    return {split: process_split(split, data_dir, out_dir) for split in ("train", "val", "test")}

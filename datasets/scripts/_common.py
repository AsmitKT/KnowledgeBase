from __future__ import annotations
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def get_dataset_config(name: str) -> dict:
    config = load_config()
    if name not in config["datasets"]:
        raise KeyError(f"Dataset config '{name}' not found in {CONFIG_PATH}")
    dataset = config["datasets"][name]
    globals_cfg = config.get("globals", {})
    result = {
        "dev_ratio": dataset.get("dev_ratio", globals_cfg.get("dev_ratio", 0.2)),
        "seed": dataset.get("seed", globals_cfg.get("seed", 42)),
        "input": {},
        "output": {}
    }
    for key, value in dataset.get("input", {}).items():
        result["input"][key] = resolve_path(value)
    for key, value in dataset.get("output", {}).items():
        result["output"][key] = resolve_path(value)
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _first_non_ws_char(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return ""
            if not ch.isspace():
                return ch


def iter_jsonl(path: Path) -> Iterator[dict]:
    first = _first_non_ws_char(path)
    if first == "[":
        raise ValueError(f"{path} looks like a JSON array, not JSONL")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")
            yield row




def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_jsonl(input_path: Path, output_path: Path, normalizer) -> int:
    ensure_dir(output_path.parent)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as out:
        for row in iter_jsonl(input_path):
            out.write(json.dumps(normalizer(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def normalize_metadata_value(value):
    if isinstance(value, dict):
        return value
    if value in (None, "", []):
        return {}
    return {"raw_metadata": value}


def normalize_corpus_record(row: dict) -> dict:
    return {
        "_id": str(row.get("_id", "")).strip(),
        "title": str(row.get("title", "")).strip(),
        "text": str(row.get("text", "")).strip(),
        "metadata": normalize_metadata_value(row.get("metadata", {}))
    }


def normalize_query_record(row: dict) -> dict:
    return {
        "_id": str(row.get("_id", "")).strip(),
        "text": str(row.get("text", "")).strip(),
        "metadata": normalize_metadata_value(row.get("metadata", {}))
    }


def read_qrels(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "\t" if "\t" in sample else None
        if delimiter is None:
            reader = csv.reader(f, delimiter=" ")
            raw_rows = []
            for row in reader:
                cleaned = [cell for cell in row if cell.strip()]
                if cleaned:
                    raw_rows.append(cleaned)
        else:
            reader = csv.reader(f, delimiter=delimiter)
            raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    for row in raw_rows:
        cells = [cell.strip() for cell in row if cell.strip()]
        if not cells:
            continue
        low = [cell.lower() for cell in cells]
        if len(cells) >= 3 and (low[0].startswith("query") or low[1].startswith("corpus") or low[2].startswith("score")):
            continue
        if len(cells) >= 3:
            qid, cid, score = cells[0], cells[1], cells[2]
        elif len(cells) == 2:
            qid, cid, score = cells[0], cells[1], "1"
        else:
            continue
        if float(score) > 0:
            rows.append((str(qid), str(cid), str(score)))
    return rows


def write_qrels(path: Path, rows: Iterable[tuple[str, str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "corpus_id", "score"])
        for qid, cid, score in rows:
            writer.writerow([qid, cid, score])


def sample_dev_from_qrels(rows: list[tuple[str, str, str]], ratio: float, seed: int) -> list[tuple[str, str, str]]:
    if not rows:
        return []
    query_ids = sorted({qid for qid, _, _ in rows})
    count = max(1, int(round(len(query_ids) * ratio)))
    count = min(count, len(query_ids))
    rng = random.Random(seed)
    selected = set(rng.sample(query_ids, count))
    return [row for row in rows if row[0] in selected]


def clean_text(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()
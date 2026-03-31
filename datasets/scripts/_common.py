from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]


def build_parser(default_input: str, default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / default_input)
    parser.add_argument("--output", type=Path, default=ROOT / default_output)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


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
        raise ValueError(f"{path} looks like a JSON array, not JSONL. Convert it to one-object-per-line first.")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}, got {type(row).__name__}")
            yield row


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def transform_jsonl(input_path: Path, output_path: Path, normalizer: Callable[[dict], dict]) -> int:
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
        "metadata": normalize_metadata_value(row.get("metadata", {})),
    }


def normalize_query_record(row: dict) -> dict:
    return {
        "_id": str(row.get("_id", "")).strip(),
        "text": str(row.get("text", "")).strip(),
        "metadata": normalize_metadata_value(row.get("metadata", {})),
    }


def read_qrels(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
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
        try:
            if float(score) <= 0:
                continue
        except ValueError:
            continue
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
    if not query_ids:
        return []
    count = max(1, int(round(len(query_ids) * ratio)))
    count = min(count, len(query_ids))
    rng = random.Random(seed)
    selected = set(rng.sample(query_ids, count))
    return [row for row in rows if row[0] in selected]


def clean_text(parts: Iterable[str]) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()

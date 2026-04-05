from __future__ import annotations
import re
from _common import clean_text, get_dataset_config, sample_dev_from_qrels, write_jsonl, write_qrels

def parse_tagged_records(path):
    records = []
    current = None
    current_field = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(".I "):
                if current is not None:
                    records.append(current)
                current = {"I": stripped[3:].strip()}
                current_field = None
                continue
            if stripped.startswith(".") and len(stripped) >= 2:
                tag = stripped[1]
                if tag in {"T", "A", "W", "X"}:
                    current_field = tag
                    current.setdefault(tag, [])
                    remainder = stripped[2:].strip()
                    if remainder:
                        current[tag].append(remainder)
                    continue
            if current is not None and current_field is not None:
                current[current_field].append(stripped)
    if current is not None:
        records.append(current)
    return records

def parse_corpus(path):
    corpus = []
    edges = []
    for record in parse_tagged_records(path):
        doc_id = str(record.get("I", "")).strip()
        title = clean_text(record.get("T", []))
        text = clean_text(record.get("W", []))
        authors = [part.strip() for part in record.get("A", []) if part.strip()]
        corpus.append({
            "_id": doc_id,
            "title": title,
            "text": text,
            "metadata": {"authors": authors}
        })
        for raw_x in record.get("X", []):
            nums = [int(x) for x in re.findall(r"-?\d+", raw_x)]
            if not nums:
                continue
            target_id = str(nums[0])
            edges.append({
                "source_id": doc_id,
                "target_id": target_id,
                "relation": "cross_reference",
                "raw": nums
            })
    return corpus, edges

def parse_queries(path):
    queries = []
    for record in parse_tagged_records(path):
        query_id = str(record.get("I", "")).strip()
        text = clean_text(record.get("W", []))
        queries.append({
            "_id": query_id,
            "text": text,
            "metadata": {}
        })
    return queries

def parse_rel(path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            parts = raw_line.strip().split()
            if len(parts) < 2:
                continue
            query_id = str(parts[0]).strip()
            corpus_id = str(parts[1]).strip()
            if query_id and corpus_id:
                rows.append((query_id, corpus_id, "1"))
    return rows

def main() -> None:
    cfg = get_dataset_config("cisi")
    corpus, edges = parse_corpus(cfg["input"]["corpus_source"])
    queries = parse_queries(cfg["input"]["queries_source"])
    test_qrels = parse_rel(cfg["input"]["qrels_source"])
    dev_qrels = sample_dev_from_qrels(test_qrels, cfg["dev_ratio"], cfg["seed"])
    write_jsonl(cfg["output"]["corpus"], corpus)
    write_jsonl(cfg["output"]["queries"], queries)
    write_jsonl(cfg["output"]["edges"], edges)
    write_qrels(cfg["output"]["dev_qrels"], dev_qrels)
    write_qrels(cfg["output"]["test_qrels"], test_qrels)

if __name__ == "__main__":
    main()
from __future__ import annotations
import re
from _common import build_parser, clean_text, read_qrels, sample_dev_from_qrels, write_jsonl, write_qrels


def parse_tagged_records(path):
    records = []
    current = None
    current_tag = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if stripped.startswith(".I"):
                if current is not None:
                    records.append(current)
                current = {"I": stripped[2:].strip()}
                current_tag = None
                continue
            if current is None:
                continue
            if stripped.startswith(".") and len(stripped) >= 2 and stripped[1].isalpha():
                current_tag = stripped[1]
                current.setdefault(current_tag, [])
                content = stripped[2:].strip()
                if content:
                    current[current_tag].append(content)
                continue
            if current_tag is not None and stripped:
                current.setdefault(current_tag, []).append(stripped)
    if current is not None:
        records.append(current)
    return records


def parse_corpus(all_path):
    corpus = []
    edges = []
    for record in parse_tagged_records(all_path):
        doc_id = str(record.get("I", "")).strip()
        title = clean_text(record.get("T", []))
        text = clean_text(record.get("W", []))
        authors = [part.strip() for part in record.get("A", []) if part.strip()]
        corpus.append({
            "_id": doc_id,
            "title": title,
            "text": text,
            "metadata": {"authors": authors},
        })
        for line in record.get("X", []):
            values = [int(value) for value in re.findall(r"-?\d+", line)]
            if not values:
                continue
            target_id = str(values[0])
            raw = values[:3] if len(values) >= 3 else values
            edges.append({
                "source_id": doc_id,
                "target_id": target_id,
                "relation": "cross_reference",
                "raw": raw,
            })
    return corpus, edges


def parse_queries(qry_path):
    queries = []
    for record in parse_tagged_records(qry_path):
        query_id = str(record.get("I", "")).strip()
        text = clean_text(record.get("W", []))
        queries.append({
            "_id": query_id,
            "text": text,
            "metadata": {},
        })
    return queries


def parse_rel(rel_path):
    rows = []
    with rel_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            values = re.findall(r"-?\d+\.?\d*", line)
            if len(values) < 2:
                continue
            qid = str(int(float(values[0])))
            cid = str(int(float(values[1])))
            rows.append((qid, cid, "1"))
    return rows


def main() -> None:
    parser = build_parser("cisi", "normalized_datasets/cisi")
    args = parser.parse_args()
    corpus, edges = parse_corpus(args.input / "CISI.ALL")
    queries = parse_queries(args.input / "CISI.QRY")
    test_qrels = parse_rel(args.input / "CISI.REL")
    dev_qrels = sample_dev_from_qrels(test_qrels, args.dev_ratio, args.seed)
    write_jsonl(args.output / "corpus.jsonl", corpus)
    write_jsonl(args.output / "queries.jsonl", queries)
    write_jsonl(args.output / "edges.jsonl", edges)
    qrels_dir = args.output / "qrels"
    write_qrels(qrels_dir / "dev.tsv", dev_qrels)
    write_qrels(qrels_dir / "test.tsv", test_qrels)


if __name__ == "__main__":
    main()

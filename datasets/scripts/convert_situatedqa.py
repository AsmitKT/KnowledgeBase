from __future__ import annotations
import json
from _common import ROOT, build_parser, write_jsonl, write_qrels


def load_split(path, split_name):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_split"] = split_name
            rows.append(row)
    return rows


def answer_to_text(value):
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip()).strip()
    return str(value).strip()


def build_geo_rows(rows):
    corpus = []
    queries = []
    dev_qrels = []
    test_qrels = []
    for index, row in enumerate(rows, start=1):
        local_id = str(index)
        question = str(row.get("question", "")).strip()
        edited_question = str(row.get("edited_question", "") or question).strip()
        answer = answer_to_text(row.get("answer", ""))
        source_id = str(row.get("id", "")).strip()
        location = str(row.get("location", "")).strip()
        corpus.append({
            "_id": local_id,
            "title": question,
            "text": f"Question: {edited_question}\nAnswer: {answer}",
            "metadata": {"location": location, "source_id": source_id},
        })
        queries.append({
            "_id": local_id,
            "text": edited_question,
            "metadata": {"location": location, "source_id": source_id},
        })
        if row["_split"] == "dev":
            dev_qrels.append((local_id, local_id, "1"))
        elif row["_split"] == "test":
            test_qrels.append((local_id, local_id, "1"))
    return corpus, queries, dev_qrels, test_qrels


def build_temp_rows(rows):
    corpus = []
    queries = []
    dev_qrels = []
    test_qrels = []
    for index, row in enumerate(rows, start=1):
        local_id = str(index)
        question = str(row.get("question", "")).strip()
        edited_question = str(row.get("edited_question", "") or question).strip()
        answer = answer_to_text(row.get("answer", ""))
        source_id = str(row.get("id", "")).strip()
        date = str(row.get("date", "")).strip()
        date_type = str(row.get("date_type", "")).strip()
        metadata = {"date": date, "source_id": source_id}
        if date_type:
            metadata["date_type"] = date_type
        corpus.append({
            "_id": local_id,
            "title": question,
            "text": f"Question: {edited_question}\nAnswer: {answer}",
            "metadata": metadata,
        })
        queries.append({
            "_id": local_id,
            "text": edited_question,
            "metadata": metadata,
        })
        if row["_split"] == "dev":
            dev_qrels.append((local_id, local_id, "1"))
        elif row["_split"] == "test":
            test_qrels.append((local_id, local_id, "1"))
    return corpus, queries, dev_qrels, test_qrels


def write_dataset(output_dir, corpus, queries, dev_qrels, test_qrels):
    write_jsonl(output_dir / "corpus.jsonl", corpus)
    write_jsonl(output_dir / "queries.jsonl", queries)
    qrels_dir = output_dir / "qrels"
    write_qrels(qrels_dir / "dev.tsv", dev_qrels)
    write_qrels(qrels_dir / "test.tsv", test_qrels)


def main() -> None:
    parser = build_parser("situatedQA/qa_data", "normalized_datasets")
    args = parser.parse_args()
    geo_rows = []
    temp_rows = []
    for split in ("train", "dev", "test"):
        geo_rows.extend(load_split(args.input / f"geo.{split}.jsonl", split))
        temp_rows.extend(load_split(args.input / f"temp.{split}.jsonl", split))
    geo_corpus, geo_queries, geo_dev, geo_test = build_geo_rows(geo_rows)
    temp_corpus, temp_queries, temp_dev, temp_test = build_temp_rows(temp_rows)
    write_dataset(args.output / "situatedqa-geo", geo_corpus, geo_queries, geo_dev, geo_test)
    write_dataset(args.output / "situatedqa-temp", temp_corpus, temp_queries, temp_dev, temp_test)


if __name__ == "__main__":
    main()

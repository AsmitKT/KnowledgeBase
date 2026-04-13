from __future__ import annotations
import json
from _common import get_dataset_config, write_jsonl, write_qrels

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
            "metadata": {"location": location, "source_id": source_id}
        })
        queries.append({
            "_id": local_id,
            "text": edited_question,
            "metadata": {"location": location, "source_id": source_id}
        })
        if row["_split"] == "dev" or row["_split"] == "test":
            test_qrels.append((local_id, local_id, "1"))
    return corpus, queries, dev_qrels, test_qrels

def build_temp_rows(rows):
    corpus = []
    queries = []
    dev_qrels = []
    test_qrels = []
    for index, row in enumerate(rows, start=1000001):
        local_id = str(index)
        question = str(row.get("question", "")).strip()
        edited_question = str(row.get("edited_question", "") or question).strip()
        answer = answer_to_text(row.get("answer", ""))
        source_id = str(row.get("id", "")).strip()
        date = str(row.get("date", "")).strip()
        date_type = str(row.get("date_type", "")).strip()
        corpus_metadata = {"date": date, "source_id": source_id}
        query_metadata = {"date": date, "source_id": source_id}
        if date_type:
            corpus_metadata["date_type"] = date_type
            query_metadata["date_type"] = date_type
        corpus.append({
            "_id": local_id,
            "title": question,
            "text": f"Question: {edited_question}\nAnswer: {answer}",
            "metadata": corpus_metadata
        })
        queries.append({
            "_id": local_id,
            "text": edited_question,
            "metadata": query_metadata
        })
        if row["_split"] == "dev" or row["_split"] == "test":
            test_qrels.append((local_id, local_id, "1"))
    return corpus, queries, dev_qrels, test_qrels

def write_dataset(corpus_path, queries_path, dev_qrels_path, test_qrels_path, corpus, queries, dev_qrels, test_qrels):
    write_jsonl(corpus_path, corpus)
    write_jsonl(queries_path, queries)
    if dev_qrels:
        write_qrels(dev_qrels_path, dev_qrels)
    elif dev_qrels_path.exists():
        dev_qrels_path.unlink()
    write_qrels(test_qrels_path, test_qrels)

def main() -> None:
    cfg = get_dataset_config("situatedqa")
    geo_rows = []
    temp_rows = []
    geo_rows.extend(load_split(cfg["input"]["geo_train"], "train"))
    geo_rows.extend(load_split(cfg["input"]["geo_dev"], "dev"))
    geo_rows.extend(load_split(cfg["input"]["geo_test"], "test"))
    temp_rows.extend(load_split(cfg["input"]["temp_train"], "train"))
    temp_rows.extend(load_split(cfg["input"]["temp_dev"], "dev"))
    temp_rows.extend(load_split(cfg["input"]["temp_test"], "test"))
    geo_corpus, geo_queries, geo_dev, geo_test = build_geo_rows(geo_rows)
    temp_corpus, temp_queries, temp_dev, temp_test = build_temp_rows(temp_rows)
    write_dataset(
        cfg["output"]["geo_corpus"],
        cfg["output"]["geo_queries"],
        cfg["output"]["geo_dev_qrels"],
        cfg["output"]["geo_test_qrels"],
        geo_corpus,
        geo_queries,
        geo_dev,
        geo_test
    )
    write_dataset(
        cfg["output"]["temp_corpus"],
        cfg["output"]["temp_queries"],
        cfg["output"]["temp_dev_qrels"],
        cfg["output"]["temp_test_qrels"],
        temp_corpus,
        temp_queries,
        temp_dev,
        temp_test
    )

if __name__ == "__main__":
    main()
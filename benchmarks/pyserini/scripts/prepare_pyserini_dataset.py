import argparse
import json
from config_paths import DATASETS, get_dataset_files, get_prepared_paths

def normalize_text(value):
    return " ".join(str(value or "").strip().split())

def load_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_qrels_query_ids(path):
    query_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:
                first = False
                if line.lower().startswith("query_id\tcorpus_id\tscore"):
                    continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            score = parts[2]
            try:
                score_val = float(score)
            except ValueError:
                continue
            if score_val > 0:
                query_ids.add(str(parts[0]))
    return query_ids

def split_question_answer(text):
    text = str(text or "").strip()
    if "Question:" in text and "Answer:" in text:
        after_question = text.split("Question:", 1)[1].strip()
        question_part, answer_part = after_question.split("Answer:", 1)
        question = normalize_text(question_part)
        answer = normalize_text(answer_part)
        return question, answer
    return normalize_text(text), ""

def flatten_metadata_value(value, prefix=""):
    parts = []
    if value is None:
        return parts
    if isinstance(value, dict):
        for key, subvalue in value.items():
            key_str = str(key).strip()
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            parts.extend(flatten_metadata_value(subvalue, next_prefix))
        return parts
    if isinstance(value, (list, tuple, set)):
        for item in value:
            parts.extend(flatten_metadata_value(item, prefix))
        return parts
    value_str = normalize_text(value)
    if not value_str:
        return parts
    if prefix:
        parts.append(f"{prefix}: {value_str}")
    else:
        parts.append(value_str)
    return parts

def metadata_to_text(metadata):
    if metadata is None:
        return ""
    if isinstance(metadata, str):
        metadata = metadata.strip()
        if not metadata:
            return ""
        try:
            metadata = json.loads(metadata)
        except Exception:
            return normalize_text(metadata)
    parts = flatten_metadata_value(metadata)
    seen = set()
    unique_parts = []
    for part in parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)
    return " | ".join(unique_parts).strip()

def parse_situated_question(dataset, question):
    question = normalize_text(question)
    if dataset == "situatedqa-geo":
        if " in " in question:
            template, context = question.rsplit(" in ", 1)
            return normalize_text(template), normalize_text(context)
        return question, ""
    if dataset == "situatedqa-temp":
        if " as of " in question:
            template, context = question.rsplit(" as of ", 1)
            return normalize_text(template), normalize_text(context)
        return question, ""
    return question, ""

def build_situated_query_text(dataset, question, metadata):
    template, context = parse_situated_question(dataset, question)
    meta_text = metadata_to_text(metadata)
    lines = []
    if template:
        lines.append(f"template: {template}")
    if dataset == "situatedqa-geo" and context:
        lines.append(f"location: {context}")
    elif dataset == "situatedqa-temp" and context:
        lines.append(f"date: {context}")
    lines.append(f"question: {question}")
    if meta_text:
        lines.append(f"metadata: {meta_text}")
    return "\n".join(lines).strip()

def build_situated_doc_text(dataset, question, answer, metadata):
    template, context = parse_situated_question(dataset, question)
    meta_text = metadata_to_text(metadata)
    lines = []
    if template:
        lines.append(f"template: {template}")
    if dataset == "situatedqa-geo" and context:
        lines.append(f"location: {context}")
    elif dataset == "situatedqa-temp" and context:
        lines.append(f"date: {context}")
    lines.append(f"question: {question}")
    if meta_text:
        lines.append(f"metadata: {meta_text}")
    if answer:
        lines.append(f"answer: {answer}")
    return "\n".join(lines).strip()

def build_standard_doc_text(row):
    title = normalize_text(row.get("title", ""))
    text = normalize_text(row.get("text", ""))
    meta_text = metadata_to_text(row.get("metadata"))
    parts = []
    if title:
        parts.append(title)
    if text:
        parts.append(text)
    if meta_text:
        parts.append(meta_text)
    return "\n".join(parts).strip()

def build_standard_query_text(row):
    text = normalize_text(row.get("text", ""))
    meta_text = metadata_to_text(row.get("metadata"))
    parts = []
    if text:
        parts.append(text)
    if meta_text:
        parts.append(meta_text)
    return "\n".join(parts).strip()

def sanitize_topic_text(text):
    text = str(text or "")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    return normalize_text(text)

def build_views(dataset, corpus_rows, query_rows, qrel_query_ids):
    corpus = {}
    queries = {}
    all_queries = {}
    if dataset.startswith("situatedqa-"):
        for row in corpus_rows:
            doc_id = str(row["_id"])
            question, answer = split_question_answer(row.get("text", ""))
            corpus[doc_id] = build_situated_doc_text(dataset, question, answer, row.get("metadata"))
        for row in query_rows:
            qid = str(row["_id"])
            question, _ = split_question_answer(row.get("text", ""))
            all_queries[qid] = build_situated_query_text(dataset, question, row.get("metadata"))
        for qid in qrel_query_ids:
            if qid in corpus:
                lines = []
                for line in corpus[qid].splitlines():
                    if not line.startswith("answer:"):
                        lines.append(line)
                queries[qid] = "\n".join(lines).strip()
            elif qid in all_queries:
                queries[qid] = all_queries[qid]
        return corpus, queries
    for row in corpus_rows:
        doc_id = str(row["_id"])
        corpus[doc_id] = build_standard_doc_text(row)
    for row in query_rows:
        qid = str(row["_id"])
        if qid in qrel_query_ids:
            queries[qid] = build_standard_query_text(row)
    return corpus, queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    args = parser.parse_args()

    files = get_dataset_files(args.dataset)
    prepared = get_prepared_paths(args.dataset)

    corpus_rows = load_jsonl_rows(files["corpus"])
    query_rows = load_jsonl_rows(files["queries"])
    qrel_query_ids = load_qrels_query_ids(files["qrels"])

    corpus, queries = build_views(args.dataset, corpus_rows, query_rows, qrel_query_ids)

    prepared["corpus_dir"].mkdir(parents=True, exist_ok=True)
    prepared["topics_dir"].mkdir(parents=True, exist_ok=True)

    with open(prepared["corpus_file"], "w", encoding="utf-8", newline="\n") as f:
        for doc_id, contents in corpus.items():
            f.write(json.dumps({"id": doc_id, "contents": contents}, ensure_ascii=False) + "\n")

    with open(prepared["topics_file"], "w", encoding="utf-8", newline="\n") as f:
        for qid, query_text in queries.items():
            f.write(f"{qid}\t{sanitize_topic_text(query_text)}\n")

if __name__ == "__main__":
    main()

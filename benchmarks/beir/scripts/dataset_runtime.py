import json
import random

from config_paths import DATASETS, get_dataset_files

HEADER = "query_id\tcorpus_id\tscore\n"
DEFAULT_RANDOM_SEED = 42

def ensure_qrels_headers(dataset_names=None):
    names = dataset_names if dataset_names is not None else DATASETS.keys()
    for dataset in names:
        dataset_path = DATASETS[dataset]
        qrels_dir = dataset_path / "qrels"
        if not qrels_dir.exists():
            continue
        for split in ("dev.tsv", "test.tsv"):
            path = qrels_dir / split
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            stripped = text.lstrip()
            if not stripped.lower().startswith("query_id\tcorpus_id\tscore"):
                path.write_text(HEADER + text, encoding="utf-8")

def load_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_qrels_tsv(path):
    qrels = {}
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
            qid, docid, score = parts[0], parts[1], parts[2]
            try:
                score_val = float(score)
            except ValueError:
                continue
            if score_val <= 0:
                continue
            if score_val.is_integer():
                rel = int(score_val)
            else:
                rel = 1
            qrels.setdefault(str(qid), {})[str(docid)] = rel
    return qrels

def normalize_text(value):
    return " ".join(str(value or "").strip().split())

def split_question_answer(text):
    text = str(text or "").strip()
    if "Question:" in text and "Answer:" in text:
        after_question = text.split("Question:", 1)[1].strip()
        question_part, answer_part = after_question.split("Answer:", 1)
        question = normalize_text(question_part)
        answer = normalize_text(answer_part)
        return question, answer
    return normalize_text(text), ""

def _flatten_metadata_value(value, prefix=""):
    parts = []

    if value is None:
        return parts

    if isinstance(value, dict):
        for key, subvalue in value.items():
            key_str = str(key).strip()
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            parts.extend(_flatten_metadata_value(subvalue, next_prefix))
        return parts

    if isinstance(value, (list, tuple, set)):
        for item in value:
            parts.extend(_flatten_metadata_value(item, prefix))
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
            parsed = json.loads(metadata)
            metadata = parsed
        except Exception:
            return normalize_text(metadata)

    parts = _flatten_metadata_value(metadata)
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

def compose_standard_text(text, metadata, inject_metadata):
    base_text = normalize_text(text)
    if not inject_metadata:
        return base_text

    meta_text = metadata_to_text(metadata)
    if not meta_text:
        return base_text

    if base_text:
        return f"metadata: {meta_text}\ntext: {base_text}"
    return f"metadata: {meta_text}"

def load_dataset(dataset: str, split: str):
    files = get_dataset_files(dataset, split)
    inject_metadata = dataset.startswith("situatedqa-")
    corpus_rows = load_jsonl_rows(files["corpus"])
    query_rows = load_jsonl_rows(files["queries"])
    qrels = load_qrels_tsv(files["qrels"])

    corpus = {}
    all_queries = {}

    if dataset.startswith("situatedqa-"):
        for row in corpus_rows:
            doc_id = str(row["_id"])
            raw_text = row.get("text", "")
            metadata = row.get("metadata")
            question, answer = split_question_answer(raw_text)

            corpus[doc_id] = {
                "title": "",
                "text": build_situated_doc_text(dataset, question, answer, metadata),
            }

        for row in query_rows:
            qid = str(row["_id"])
            raw_text = row.get("text", "")
            metadata = row.get("metadata")
            question, _ = split_question_answer(raw_text)
            all_queries[qid] = build_situated_query_text(dataset, question, metadata)

        queries = {}
        for qid in qrels.keys():
            if qid in corpus:
                corpus_question_text = corpus[qid]["text"]
                lines = []
                for line in corpus_question_text.splitlines():
                    if not line.startswith("answer:"):
                        lines.append(line)
                queries[qid] = "\n".join(lines).strip()
            elif qid in all_queries:
                queries[qid] = all_queries[qid]

        qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}
        return corpus, queries, qrels

    for row in corpus_rows:
        doc_id = str(row["_id"])
        title = normalize_text(row.get("title", ""))
        text = compose_standard_text(row.get("text", ""), row.get("metadata"), inject_metadata)
        corpus[doc_id] = {
            "title": title,
            "text": text,
        }

    for row in query_rows:
        qid = str(row["_id"])
        text = compose_standard_text(row.get("text", ""), row.get("metadata"), inject_metadata)
        all_queries[qid] = text

    queries = {qid: all_queries[qid] for qid in qrels.keys() if qid in all_queries}
    qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}

    return corpus, queries, qrels

def sample_corpus_and_align(corpus, queries, qrels, size_percent: float, seed: int = DEFAULT_RANDOM_SEED):
    original_doc_count = len(corpus)

    if size_percent >= 100.0:
        meta = {
            "size_percent": 100.0,
            "random_seed": seed,
            "original_num_docs": original_doc_count,
            "sampled_num_docs": original_doc_count,
            "num_queries": len(queries),
            "num_qrels_queries": len(qrels),
        }
        return corpus, queries, qrels, meta

    if size_percent <= 0.0:
        raise ValueError("size must be > 0")

    doc_ids = list(corpus.keys())
    sample_count = max(1, int(len(doc_ids) * size_percent / 100.0))
    rng = random.Random(seed)
    selected_doc_ids = set(rng.sample(doc_ids, sample_count))

    sampled_corpus = {doc_id: corpus[doc_id] for doc_id in doc_ids if doc_id in selected_doc_ids}

    sampled_qrels = {}
    for qid, rels in qrels.items():
        kept_rels = {doc_id: score for doc_id, score in rels.items() if doc_id in selected_doc_ids}
        if kept_rels:
            sampled_qrels[qid] = kept_rels

    sampled_queries = {qid: queries[qid] for qid in sampled_qrels.keys() if qid in queries}

    meta = {
        "size_percent": size_percent,
        "random_seed": seed,
        "original_num_docs": original_doc_count,
        "sampled_num_docs": len(sampled_corpus),
        "num_queries": len(sampled_queries),
        "num_qrels_queries": len(sampled_qrels),
    }

    return sampled_corpus, sampled_queries, sampled_qrels, meta

def prepare_dataset(dataset: str, split: str, size_percent: float = 100.0, fix_qrels_headers: bool = False, seed: int = DEFAULT_RANDOM_SEED):
    if fix_qrels_headers:
        ensure_qrels_headers([dataset])

    corpus, queries, qrels = load_dataset(dataset, split)
    corpus, queries, qrels, meta = sample_corpus_and_align(corpus, queries, qrels, size_percent, seed)

    if len(corpus) == 0:
        raise ValueError("No corpus documents remain after sampling.")
    if len(queries) == 0:
        raise ValueError("No queries remain after aligning qrels with sampled corpus.")
    if len(qrels) == 0:
        raise ValueError("No qrels remain after aligning qrels with sampled corpus.")

    return corpus, queries, qrels, meta
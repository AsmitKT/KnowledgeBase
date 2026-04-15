import argparse
import json
from pathlib import Path

import numpy as np

from beir import util
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval

from config_paths import BENCH_ROOT, DATASETS
from dataset_runtime import DEFAULT_RANDOM_SEED, prepare_dataset

DEFAULT_MODELS = {
    "sentencebert": "Alibaba-NLP/gte-modernbert-base",
    "huggingface": "intfloat/e5-base-v2",
}

def build_encoder(backend: str, model_name: str):
    if backend == "sentencebert":
        return models.SentenceBERT(model_name)
    if backend == "huggingface":
        return models.HuggingFace(
            model_path=model_name,
            max_length=512,
            pooling="mean",
            normalize=True,
            prompts={"query": "query: ", "passage": "passage: "},
        )
    raise ValueError(f"Unsupported dense backend: {backend}")

def save_runfile_utf8(path: Path, results, run_name: str = "beir"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docs in results.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            for doc_id, score in ranked:
                f.write(f"{str(qid)} Q0 {str(doc_id)} 0 {score} {run_name}\n")

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def to_numpy_float32(array) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def encode_corpus_embeddings(encoder, documents, batch_size: int) -> np.ndarray:
    try:
        embeddings = encoder.encode_corpus(documents, batch_size=batch_size)
    except TypeError:
        embeddings = encoder.encode_corpus(documents)
    return to_numpy_float32(embeddings)

def encode_query_embeddings(encoder, query_texts, batch_size: int) -> np.ndarray:
    try:
        embeddings = encoder.encode_queries(query_texts, batch_size=batch_size)
    except TypeError:
        embeddings = encoder.encode_queries(query_texts)
    return to_numpy_float32(embeddings)

def build_hnsw_index(corpus_embeddings: np.ndarray, m: int, ef_construction: int, ef_search: int, num_threads: int, random_seed: int):
    try:
        import hnswlib
    except ImportError as exc:
        raise ImportError("hnswlib is required for run_dense_hnsw_beir.py. Install it with: pip install hnswlib") from exc

    if corpus_embeddings.ndim != 2:
        raise ValueError("Corpus embeddings must be a 2D array.")

    dim = int(corpus_embeddings.shape[1])
    count = int(corpus_embeddings.shape[0])
    index_ids = np.arange(count, dtype=np.int64)

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=count, ef_construction=ef_construction, M=m, random_seed=random_seed)
    index.set_num_threads(num_threads)
    index.add_items(corpus_embeddings, index_ids)
    index.set_ef(max(int(ef_search), 1))
    return index

def hnsw_search(index, query_embeddings: np.ndarray, doc_ids, top_k: int):
    if query_embeddings.ndim != 2:
        raise ValueError("Query embeddings must be a 2D array.")

    local_top_k = min(int(top_k), len(doc_ids))
    labels, distances = index.knn_query(query_embeddings, k=local_top_k)

    results = {}
    for row_idx, doc_row in enumerate(labels):
        row_results = {}
        for col_idx, internal_id in enumerate(doc_row):
            internal_id = int(internal_id)
            if internal_id < 0:
                continue
            score = 1.0 - float(distances[row_idx][col_idx])
            if score > 1.0:
                score = 1.0
            if score < -1.0:
                score = -1.0
            row_results[doc_ids[internal_id]] = score
        results[row_idx] = row_results
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--backend", required=True, choices=["sentencebert", "huggingface"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--size", type=float, default=100.0)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--save_index", action="store_true")
    args = parser.parse_args()

    split = "test"
    evaluator = EvaluateRetrieval(None)
    required_top_k = max(evaluator.k_values)
    if args.top_k < required_top_k:
        raise ValueError(f"top_k must be >= {required_top_k} to support the default BEIR metrics.")

    model_name = args.model_name or DEFAULT_MODELS[args.backend]
    corpus, queries, qrels, meta = prepare_dataset(
        dataset=args.dataset,
        size_percent=args.size,
        fix_qrels_headers=False,
        seed=args.random_seed,
    )

    encoder = build_encoder(args.backend, model_name)

    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    documents = [corpus[doc_id] for doc_id in doc_ids]
    query_texts = [queries[qid] for qid in query_ids]

    corpus_embeddings = encode_corpus_embeddings(encoder, documents, args.batch_size)
    query_embeddings = encode_query_embeddings(encoder, query_texts, args.batch_size)

    if len(doc_ids) != corpus_embeddings.shape[0]:
        raise ValueError("Mismatch between corpus documents and corpus embeddings.")
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("Mismatch between queries and query embeddings.")

    if args.ef_search < args.top_k:
        args.ef_search = args.top_k

    index = build_hnsw_index(
        corpus_embeddings=corpus_embeddings,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        num_threads=args.num_threads,
        random_seed=args.random_seed,
    )

    raw_results = hnsw_search(index, query_embeddings, doc_ids, args.top_k)
    results = {query_ids[row_idx]: docs for row_idx, docs in raw_results.items() if query_ids[row_idx] in qrels}

    ignore_identical = not args.dataset.startswith("situatedqa-")
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        results,
        evaluator.k_values,
        ignore_identical_ids=ignore_identical,
    )
    mrr = evaluator.evaluate_custom(qrels, results, evaluator.k_values, metric="mrr")

    run_dir = BENCH_ROOT / "runs" / "dense_hnsw" / args.backend / args.dataset
    result_dir = BENCH_ROOT / "results" / "dense_hnsw" / args.backend / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / f"{split}.results.json", results)
    save_runfile_utf8(run_dir / f"{split}.run.trec", results)
    util.save_results(str(result_dir / f"{split}.json"), ndcg, _map, recall, precision, mrr)

    index_path = None
    doc_ids_path = None
    if args.save_index:
        index_path = run_dir / f"{split}.hnswlib.index"
        doc_ids_path = run_dir / f"{split}.doc_ids.json"
        index.save_index(str(index_path))
        save_json(doc_ids_path, doc_ids)

    summary = {
        "family": "dense_hnsw",
        "backend": args.backend,
        "model_name": model_name,
        "dataset": args.dataset,
        "split": split,
        "batch_size": args.batch_size,
        "size_percent": meta["size_percent"],
        "random_seed": args.random_seed,
        "original_num_docs": meta["original_num_docs"],
        "sampled_num_docs": meta["sampled_num_docs"],
        "num_queries": meta["num_queries"],
        "num_qrels_queries": meta["num_qrels_queries"],
        "top_k": args.top_k,
        "space": "cosine",
        "M": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "num_threads": args.num_threads,
        "index_path": str(index_path) if index_path is not None else None,
        "doc_ids_path": str(doc_ids_path) if doc_ids_path is not None else None,
        "NDCG@10": ndcg.get("NDCG@10"),
        "MAP@100": _map.get("MAP@100"),
        "Recall@100": recall.get("Recall@100"),
        "MRR@10": mrr.get("MRR@10"),
        "P@10": precision.get("P@10"),
    }
    save_json(result_dir / f"{split}.summary.json", summary)

if __name__ == "__main__":
    main()
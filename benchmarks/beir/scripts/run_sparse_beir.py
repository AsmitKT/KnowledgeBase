import argparse
import json
from pathlib import Path

import numpy as np

if "int" not in np.__dict__:
    np.int = int
if "float" not in np.__dict__:
    np.float = float
if "bool" not in np.__dict__:
    np.bool = bool
if "complex" not in np.__dict__:
    np.complex = complex
if "object" not in np.__dict__:
    np.object = object
if "str" not in np.__dict__:
    np.str = str

from beir import util
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models.unicoil import UniCoilEncoder
from beir.retrieval.search import BaseSearch

from config_paths import BENCH_ROOT, DATASETS
from dataset_runtime import DEFAULT_RANDOM_SEED, prepare_dataset
from retrieval_runtime import run_full_or_chunked_search

DEFAULT_MODELS = {
    "sparta": "BeIR/sparta-msmarco-distilbert-base-v1",
    "splade": "naver/splade-cocondenser-ensembledistil",
    "unicoil": "castorini/unicoil-msmarco-passage",
}

QUERY_WEIGHTS = {
    "sparta": False,
    "splade": True,
    "unicoil": True,
}

INTERNAL_CHUNK_DOCS = 10000

class LocalSparseSearch(BaseSearch):
    def __init__(self, model, batch_size: int = 16, query_weights: bool = False, chunk_docs: int = INTERNAL_CHUNK_DOCS):
        self.model = model
        self.batch_size = batch_size
        self.query_weights = query_weights
        self.chunk_docs = chunk_docs
        self.results = {}
        self.last_search_mode = None
        self.last_chunk_docs_used = None

    def _encode_corpus(self, documents):
        return self.model.encode_corpus(documents, batch_size=self.batch_size)

    def _encode_single_query(self, query_text):
        if hasattr(self.model, "encode_query"):
            return self.model.encode_query(query_text)
        if hasattr(self.model, "encode_queries"):
            query_vec = self.model.encode_queries([query_text], batch_size=self.batch_size)
            return query_vec[0]
        raise AttributeError(f"{self.model.__class__.__name__} has neither encode_query nor encode_queries")

    def _score_chunk(self, corpus_repr, query_repr):
        if self.query_weights:
            scores = corpus_repr.dot(query_repr)
            scores = np.asarray(scores).squeeze()
        else:
            scores = np.asarray(corpus_repr[query_repr, :].sum(axis=0)).squeeze(0)
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        return scores

    def search(self, corpus, queries, top_k, score_function=None, **kwargs):
        self.results, self.last_search_mode, self.last_chunk_docs_used = run_full_or_chunked_search(
            corpus=corpus,
            queries=queries,
            top_k=top_k,
            chunk_docs=self.chunk_docs,
            encode_corpus_fn=self._encode_corpus,
            encode_query_fn=self._encode_single_query,
            score_chunk_fn=self._score_chunk,
            exclude_self=False,
        )
        return self.results

    def encode(self, corpus, queries, encode_output_path="./embeddings/", overwrite=False, query_filename="queries.pkl", corpus_filename="corpus.*.pkl", **kwargs):
        raise NotImplementedError("LocalSparseSearch does not implement file-based encoding.")

    def search_from_files(self, query_embeddings_file, corpus_embeddings_files, top_k, **kwargs):
        raise NotImplementedError("LocalSparseSearch does not implement file-based sparse search.")

def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_runfile_utf8(path: Path, results, run_name: str = "beir"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for qid, docs in results.items():
            ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)
            for doc_id, score in ranked:
                f.write(f"{str(qid)} Q0 {str(doc_id)} 0 {score} {run_name}\n")

def patch_unicoil_encoder():
    if not hasattr(UniCoilEncoder, "all_tied_weights_keys"):
        UniCoilEncoder.all_tied_weights_keys = {}

def build_model(backend: str, model_name: str, batch_size: int):
    if backend == "sparta":
        return LocalSparseSearch(models.SPARTA(model_name), batch_size=batch_size, query_weights=QUERY_WEIGHTS[backend], chunk_docs=INTERNAL_CHUNK_DOCS)
    if backend == "splade":
        return LocalSparseSearch(models.SPLADE(model_name), batch_size=batch_size, query_weights=QUERY_WEIGHTS[backend], chunk_docs=INTERNAL_CHUNK_DOCS)
    if backend == "unicoil":
        patch_unicoil_encoder()
        return LocalSparseSearch(models.UniCOIL(model_name), batch_size=batch_size, query_weights=QUERY_WEIGHTS[backend], chunk_docs=INTERNAL_CHUNK_DOCS)
    raise ValueError(f"Unsupported sparse backend: {backend}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--backend", required=True, choices=["sparta", "splade", "unicoil"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--size", type=float, default=100.0)
    args = parser.parse_args()

    split = "test"
    model_name = DEFAULT_MODELS[args.backend]
    corpus, queries, qrels, meta = prepare_dataset(
        dataset=args.dataset,
        size_percent=args.size,
        fix_qrels_headers=False,
        seed=DEFAULT_RANDOM_SEED,
    )

    searcher = build_model(args.backend, model_name, args.batch_size)
    retriever = EvaluateRetrieval(searcher)
    results = retriever.retrieve(corpus, queries)
    results = {qid: docs for qid, docs in results.items() if qid in qrels}

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    run_dir = BENCH_ROOT / "runs" / "sparse" / args.backend / args.dataset
    result_dir = BENCH_ROOT / "results" / "sparse" / args.backend / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / f"{split}.results.json", results)
    save_json(result_dir / f"{split}.summary.json", {
        "family": "sparse",
        "backend": args.backend,
        "model_name": model_name,
        "dataset": args.dataset,
        "split": split,
        "batch_size": args.batch_size,
        "size_percent": meta["size_percent"],
        "random_seed": meta["random_seed"],
        "original_num_docs": meta["original_num_docs"],
        "sampled_num_docs": meta["sampled_num_docs"],
        "num_queries": meta["num_queries"],
        "num_qrels_queries": meta["num_qrels_queries"],
        "search_mode": searcher.last_search_mode,
        "chunk_docs_used": searcher.last_chunk_docs_used,
        "NDCG@10": ndcg.get("NDCG@10"),
        "MAP@100": _map.get("MAP@100"),
        "Recall@100": recall.get("Recall@100"),
        "MRR@10": mrr.get("MRR@10"),
        "P@10": precision.get("P@10"),
    })

    save_runfile_utf8(run_dir / f"{split}.run.trec", results)
    util.save_results(str(result_dir / f"{split}.json"), ndcg, _map, recall, precision, mrr)

if __name__ == "__main__":
    main()
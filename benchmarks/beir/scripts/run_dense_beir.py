import argparse
import json
from pathlib import Path

from beir import util
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from config_paths import BENCH_ROOT, DATASETS
from dataset_runtime import DEFAULT_RANDOM_SEED, prepare_dataset

DEFAULT_MODELS = {
    "sentencebert": "Alibaba-NLP/gte-modernbert-base",
    "huggingface": "intfloat/e5-base-v2",
}

def build_model(backend: str, model_name: str, batch_size: int):
    if backend == "sentencebert":
        return DRES(models.SentenceBERT(model_name), batch_size=batch_size)
    if backend == "huggingface":
        return DRES(
            models.HuggingFace(
                model_path=model_name,
                max_length=512,
                pooling="mean",
                normalize=True,
                prompts={"query": "query: ", "passage": "passage: "},
            ),
            batch_size=batch_size,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--backend", required=True, choices=["sentencebert", "huggingface"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--size", type=float, default=100.0)
    args = parser.parse_args()

    split = "test"
    model_name = args.model_name or DEFAULT_MODELS[args.backend]
    corpus, queries, qrels, meta = prepare_dataset(
        dataset=args.dataset,
        size_percent=args.size,
        fix_qrels_headers=False,
        seed=DEFAULT_RANDOM_SEED,
    )

    model = build_model(args.backend, model_name, args.batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    results = {qid: docs for qid, docs in results.items() if qid in qrels}

    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    run_dir = BENCH_ROOT / "runs" / "dense" / args.backend / args.dataset
    result_dir = BENCH_ROOT / "results" / "dense" / args.backend / args.dataset
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / f"{split}.results.json", results)
    summary = {
        "family": "dense",
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
        "NDCG@10": ndcg.get("NDCG@10"),
        "MAP@100": _map.get("MAP@100"),
        "Recall@100": recall.get("Recall@100"),
        "MRR@10": mrr.get("MRR@10"),
        "P@10": precision.get("P@10"),
    }
    save_json(result_dir / f"{split}.summary.json", summary)

    save_runfile_utf8(run_dir / f"{split}.run.trec", results)
    util.save_results(str(result_dir / f"{split}.json"), ndcg, _map, recall, precision, mrr)

if __name__ == "__main__":
    main()
import argparse
import json
from beir import util
from beir.retrieval.evaluation import EvaluateRetrieval
from config_paths import DATASETS, get_dataset_files, get_result_json_path, get_result_summary_path, get_run_path, get_run_results_path

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
            qrels.setdefault(str(qid), {})[str(docid)] = int(score_val) if score_val.is_integer() else 1
    return qrels

def load_trec_run(path):
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = str(parts[0])
            docid = str(parts[2])
            score = float(parts[4])
            results.setdefault(qid, {})[docid] = score
    return results

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    args = parser.parse_args()

    files = get_dataset_files(args.dataset)
    qrels = load_qrels_tsv(files["qrels"])
    run_path = get_run_path(args.dataset)
    results = load_trec_run(run_path)

    if not qrels:
        raise ValueError(f"No positive qrels found for {args.dataset}")
    if not results:
        raise ValueError(f"No run results found at {run_path}")

    overlap_qids = sorted(set(qrels.keys()) & set(results.keys()))
    if not overlap_qids:
        raise ValueError(f"No overlapping query ids between qrels and run for {args.dataset}")

    aligned_qrels = {qid: qrels[qid] for qid in overlap_qids}
    aligned_results = {qid: results[qid] for qid in overlap_qids if results[qid]}
    if not aligned_results:
        raise ValueError(f"All overlapping queries have empty result lists for {args.dataset}")

    evaluator = EvaluateRetrieval(None)
    ignore_identical = not args.dataset.startswith("situatedqa-")
    ndcg, _map, recall, precision = evaluator.evaluate(
        aligned_qrels,
        aligned_results,
        evaluator.k_values,
        ignore_identical_ids=ignore_identical,
    )
    mrr = evaluator.evaluate_custom(aligned_qrels, aligned_results, evaluator.k_values, metric="mrr")

    hit_queries = 0
    for qid, docs in aligned_results.items():
        if any(docid in aligned_qrels[qid] for docid in docs):
            hit_queries += 1

    run_results_path = get_run_results_path(args.dataset)
    result_json_path = get_result_json_path(args.dataset)
    result_summary_path = get_result_summary_path(args.dataset)

    save_json(run_results_path, aligned_results)
    result_json_path.parent.mkdir(parents=True, exist_ok=True)
    util.save_results(str(result_json_path), ndcg, _map, recall, precision, mrr)

    summary = {
        "family": "bm25",
        "backend": "pyserini",
        "dataset": args.dataset,
        "split": "test",
        "num_qrels_queries": len(qrels),
        "num_run_queries": len(results),
        "num_overlap_queries": len(overlap_qids),
        "num_nonempty_overlap_queries": len(aligned_results),
        "num_queries_with_any_relevant_hit": hit_queries,
        "NDCG@10": ndcg.get("NDCG@10"),
        "MAP@100": _map.get("MAP@100"),
        "Recall@100": recall.get("Recall@100"),
        "MRR@10": mrr.get("MRR@10"),
        "P@10": precision.get("P@10"),
    }
    save_json(result_summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary written to: {result_summary_path}")

if __name__ == "__main__":
    main()

from pathlib import Path

BENCH_ROOT = Path(r"C:\Coding\KnowledgeBase\benchmarks\beir")
DATASET_ROOT = Path(r"C:\Coding\KnowledgeBase\datasets\normalized_datasets")

DATASETS = {
    "cisi": DATASET_ROOT / "cisi",
    "dbpedia-entity": DATASET_ROOT / "dbpedia-entity",
    "nq": DATASET_ROOT / "nq",
    "scifact": DATASET_ROOT / "scifact",
    "situatedqa-geo": DATASET_ROOT / "situatedqa-geo",
    "situatedqa-temp": DATASET_ROOT / "situatedqa-temp",
}

def get_dataset_files(dataset: str, split: str):
    base = DATASETS[dataset]
    return {
        "base": base,
        "corpus": base / "corpus.jsonl",
        "queries": base / "queries.jsonl",
        "qrels": base / "qrels" / f"{split}.tsv",
    }
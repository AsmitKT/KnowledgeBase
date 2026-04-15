import argparse
import subprocess
import sys
from config_paths import DATASETS

DEFAULT_ORDER = [
    "scifact",
    "nq",
    "dbpedia-entity",
    "cisi",
    "situatedqa-geo",
    "situatedqa-temp",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    datasets = [dataset for dataset in DEFAULT_ORDER if dataset in DATASETS]

    for dataset in datasets:
        subprocess.run([
            sys.executable,
            "benchmarks/pyserini/scripts/prepare_pyserini_dataset.py",
            "--dataset",
            dataset,
        ], check=True)
        run_cmd = [
            sys.executable,
            "benchmarks/pyserini/scripts/run_bm25_pyserini.py",
            "--dataset", dataset,
            "--threads", str(args.threads),
            "--hits", str(args.hits),
            "--batch_size", str(args.batch_size),
        ]
        if args.reindex:
            run_cmd.append("--reindex")
        subprocess.run(run_cmd, check=True)
        subprocess.run([
            sys.executable,
            "benchmarks/pyserini/scripts/evaluate_bm25_pyserini.py",
            "--dataset",
            dataset,
        ], check=True)

if __name__ == "__main__":
    main()

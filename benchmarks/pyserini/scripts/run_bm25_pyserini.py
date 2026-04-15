import argparse
import shutil
import subprocess
import sys
from config_paths import DATASETS, get_index_path, get_prepared_paths, get_run_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hits", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    prepared = get_prepared_paths(args.dataset)
    index_path = get_index_path(args.dataset)
    run_path = get_run_path(args.dataset)

    if not prepared["corpus_file"].exists() or not prepared["topics_file"].exists():
        raise FileNotFoundError("Prepared corpus/topics not found. Run prepare_pyserini_dataset.py first.")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reindex and index_path.exists():
        shutil.rmtree(index_path)

    if args.reindex or not index_path.exists():
        index_cmd = [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(prepared["corpus_dir"]),
            "--index", str(index_path),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", str(args.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ]
        subprocess.run(index_cmd, check=True)

    search_cmd = [
        sys.executable,
        "-m",
        "pyserini.search.lucene",
        "--index", str(index_path),
        "--topics", str(prepared["topics_file"]),
        "--output", str(run_path),
        "--bm25",
        "--hits", str(args.hits),
        "--threads", str(args.threads),
        "--batch-size", str(args.batch_size),
        "--k1", str(args.k1),
        "--b", str(args.b),
    ]
    subprocess.run(search_cmd, check=True)

if __name__ == "__main__":
    main()

from pathlib import Path
import json

CONFIG_FILENAME = "datasets_config.json"

def _find_in_parents(filename: str, start_dir: Path) -> Path:
    start_dir = start_dir.resolve()
    for base in (start_dir, *start_dir.parents):
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {filename} starting from {start_dir}")

def _resolve_from_project_root(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

def _derive_datasets(config: dict, project_root: Path):
    if "beir_datasets" in config:
        return {
            name: _resolve_from_project_root(project_root, dataset_cfg["dir"])
            for name, dataset_cfg in config["beir_datasets"].items()
        }

    datasets_cfg = config.get("datasets", {})
    derived = {}

    if "nq" in datasets_cfg:
        derived["nq"] = _resolve_from_project_root(project_root, str(Path(datasets_cfg["nq"]["output"]["corpus"]).parent))

    if "scifact" in datasets_cfg:
        derived["scifact"] = _resolve_from_project_root(project_root, str(Path(datasets_cfg["scifact"]["output"]["corpus"]).parent))

    if "dbpedia_entity" in datasets_cfg:
        derived["dbpedia-entity"] = _resolve_from_project_root(project_root, str(Path(datasets_cfg["dbpedia_entity"]["output"]["corpus"]).parent))

    if "cisi" in datasets_cfg:
        derived["cisi"] = _resolve_from_project_root(project_root, str(Path(datasets_cfg["cisi"]["output"]["corpus"]).parent))

    if "situatedqa" in datasets_cfg:
        out = datasets_cfg["situatedqa"]["output"]
        derived["situatedqa-geo"] = _resolve_from_project_root(project_root, str(Path(out["geo_corpus"]).parent))
        derived["situatedqa-temp"] = _resolve_from_project_root(project_root, str(Path(out["temp_corpus"]).parent))

    return derived

_SCRIPT_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _find_in_parents(CONFIG_FILENAME, _SCRIPT_DIR)
_PROJECT_ROOT = _CONFIG_PATH.parent

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _CONFIG = json.load(f)

BENCH_ROOT = (_PROJECT_ROOT / "benchmarks" / "pyserini").resolve()
DATASETS = _derive_datasets(_CONFIG, _PROJECT_ROOT)

def get_dataset_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASETS[dataset]

def get_dataset_files(dataset: str):
    base = get_dataset_dir(dataset)
    return {
        "base": base,
        "corpus": base / "corpus.jsonl",
        "queries": base / "queries.jsonl",
        "qrels": base / "qrels" / "test.tsv",
    }

def get_prepared_paths(dataset: str):
    base = BENCH_ROOT / "prepared" / dataset
    return {
        "base": base,
        "corpus_dir": base / "corpus",
        "corpus_file": base / "corpus" / "part-00000.jsonl",
        "topics_dir": base / "topics",
        "topics_file": base / "topics" / "test.tsv",
    }

def get_index_path(dataset: str):
    return BENCH_ROOT / "indexes" / "bm25" / "lucene" / dataset

def get_run_path(dataset: str):
    return BENCH_ROOT / "runs" / "bm25" / "pyserini" / dataset / "test.run.trec"

def get_run_results_path(dataset: str):
    return BENCH_ROOT / "runs" / "bm25" / "pyserini" / dataset / "test.results.json"

def get_result_json_path(dataset: str):
    return BENCH_ROOT / "results" / "bm25" / "pyserini" / dataset / "test.json"

def get_result_summary_path(dataset: str):
    return BENCH_ROOT / "results" / "bm25" / "pyserini" / dataset / "test.summary.json"
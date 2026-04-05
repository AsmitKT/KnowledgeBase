from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from _common import load_config

ROOT = Path(__file__).resolve().parent

SCRIPT_BY_DATASET = {
    "nq": "convert_nq.py",
    "scifact": "convert_scifact.py",
    "dbpedia_entity": "convert_dbpedia_entity.py",
    "cisi": "convert_cisi.py",
    "situatedqa": "convert_situatedqa.py"
}


def main() -> None:
    config = load_config()
    for dataset_name in config["run_order"]:
        script = SCRIPT_BY_DATASET[dataset_name]
        print(f"starting {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], check=True)
        print(f"finished {script}", flush=True)


if __name__ == "__main__":
    main()
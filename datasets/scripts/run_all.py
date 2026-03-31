from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    "convert_nq.py",
    "convert_scifact.py",
    "convert_dbpedia_entity.py",
    "convert_cisi.py",
    "convert_situatedqa.py",
]

def main() -> None:
    for script in SCRIPTS:
        print(f"starting {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], check=True)
        print(f"finished {script}", flush=True)

if __name__ == "__main__":
    main()
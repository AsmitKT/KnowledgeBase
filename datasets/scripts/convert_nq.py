from __future__ import annotations
from _common import copy_file, get_dataset_config, read_qrels, write_qrels

def main() -> None:
    cfg = get_dataset_config("nq")
    test_qrels = read_qrels(cfg["input"]["test_qrels"])
    copy_file(cfg["input"]["corpus"], cfg["output"]["corpus"])
    copy_file(cfg["input"]["queries"], cfg["output"]["queries"])
    write_qrels(cfg["output"]["test_qrels"], test_qrels)

if __name__ == "__main__":
    main()
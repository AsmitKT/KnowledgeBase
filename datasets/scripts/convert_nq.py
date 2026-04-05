from __future__ import annotations
from _common import copy_file, get_dataset_config, read_qrels, sample_dev_from_qrels, write_qrels

def main() -> None:
    cfg = get_dataset_config("nq")
    test_qrels = read_qrels(cfg["input"]["test_qrels"])
    dev_qrels = sample_dev_from_qrels(test_qrels, cfg["dev_ratio"], cfg["seed"])
    copy_file(cfg["input"]["corpus"], cfg["output"]["corpus"])
    copy_file(cfg["input"]["queries"], cfg["output"]["queries"])
    write_qrels(cfg["output"]["dev_qrels"], dev_qrels)
    write_qrels(cfg["output"]["test_qrels"], test_qrels)

if __name__ == "__main__":
    main()
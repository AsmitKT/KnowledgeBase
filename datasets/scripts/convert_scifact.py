from __future__ import annotations
from _common import get_dataset_config, normalize_corpus_record, normalize_query_record, read_qrels, transform_jsonl, write_qrels


def main() -> None:
    cfg = get_dataset_config("scifact")
    dev_qrels = read_qrels(cfg["input"]["source_dev_qrels"])
    test_qrels = read_qrels(cfg["input"]["source_test_qrels"])
    transform_jsonl(cfg["input"]["corpus"], cfg["output"]["corpus"], normalize_corpus_record)
    transform_jsonl(cfg["input"]["queries"], cfg["output"]["queries"], normalize_query_record)
    write_qrels(cfg["output"]["dev_qrels"], dev_qrels)
    write_qrels(cfg["output"]["test_qrels"], test_qrels)


if __name__ == "__main__":
    main()
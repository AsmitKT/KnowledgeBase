from __future__ import annotations
from _common import get_dataset_config, normalize_corpus_record, normalize_query_record, read_qrels, transform_jsonl, write_qrels


def main() -> None:
    cfg = get_dataset_config("dbpedia_entity")
    dev_qrels = read_qrels(cfg["input"]["dev_qrels"])
    test_qrels = read_qrels(cfg["input"]["test_qrels"])
    merged_test_qrels = dev_qrels + test_qrels
    transform_jsonl(cfg["input"]["corpus"], cfg["output"]["corpus"], normalize_corpus_record)
    transform_jsonl(cfg["input"]["queries"], cfg["output"]["queries"], normalize_query_record)
    write_qrels(cfg["output"]["test_qrels"], merged_test_qrels)

if __name__ == "__main__":
    main()
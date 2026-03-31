from __future__ import annotations
from _common import build_parser, normalize_corpus_record, normalize_query_record, read_qrels, sample_dev_from_qrels, transform_jsonl, write_qrels


def main() -> None:
    parser = build_parser("nq", "normalized_datasets/nq")
    args = parser.parse_args()
    test_qrels = read_qrels(args.input / "qrels" / "test.tsv")
    dev_qrels = sample_dev_from_qrels(test_qrels, args.dev_ratio, args.seed)
    transform_jsonl(args.input / "corpus.jsonl", args.output / "corpus.jsonl", normalize_corpus_record)
    transform_jsonl(args.input / "queries.jsonl", args.output / "queries.jsonl", normalize_query_record)
    qrels_dir = args.output / "qrels"
    write_qrels(qrels_dir / "dev.tsv", dev_qrels)
    write_qrels(qrels_dir / "test.tsv", test_qrels)

if __name__ == "__main__":
    main()

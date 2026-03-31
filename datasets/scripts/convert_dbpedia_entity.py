from __future__ import annotations
from _common import build_parser, normalize_corpus_record, normalize_query_record, read_qrels, transform_jsonl, write_qrels


def main() -> None:
    parser = build_parser("dbpedia-entity", "normalized_datasets/dbpedia-entity")
    args = parser.parse_args()
    transform_jsonl(args.input / "corpus.jsonl", args.output / "corpus.jsonl", normalize_corpus_record)
    transform_jsonl(args.input / "queries.jsonl", args.output / "queries.jsonl", normalize_query_record)
    dev_qrels = read_qrels(args.input / "qrels" / "dev.tsv")
    test_qrels = read_qrels(args.input / "qrels" / "test.tsv")
    qrels_dir = args.output / "qrels"
    write_qrels(qrels_dir / "dev.tsv", dev_qrels)
    write_qrels(qrels_dir / "test.tsv", test_qrels)


if __name__ == "__main__":
    main()

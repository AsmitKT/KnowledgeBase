import json
import os
import random

DEFAULT_RANDOM_SEED=42

def load_jsonl(path):
    data=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_qrels(path):
    qrels={}
    if not os.path.exists(path):
        return qrels

    with open(path,'r',encoding='utf-8') as f:
        first=True
        for line in f:
            line=line.strip()
            if not line:
                continue

            if first:
                first=False
                if line.lower().startswith("query_id\tcorpus_id\tscore"):
                    continue

            parts=line.split('\t')
            if len(parts)<3:
                continue

            qid=str(parts[0])
            doc_id=str(parts[1])

            try:
                score=float(parts[2])
            except ValueError:
                continue

            if score<=0:
                continue

            qrels.setdefault(qid,set()).add(doc_id)

    return qrels

def _get_row_id(row,keys):
    for key in keys:
        value=row.get(key)
        if value is not None:
            return str(value)
    return None

def sample_corpus_and_align(corpus,queries,qrels,size_percent,seed=DEFAULT_RANDOM_SEED):
    original_doc_count=len(corpus)

    if size_percent>=100.0:
        return corpus,queries,qrels,{
            "size_percent":100.0,
            "random_seed":seed,
            "original_num_docs":original_doc_count,
            "sampled_num_docs":original_doc_count,
            "num_queries":len(queries),
            "num_qrels_queries":len(qrels)
        }

    if size_percent<=0.0:
        raise ValueError("size must be > 0")

    corpus_ids=[]
    corpus_by_id={}
    for row in corpus:
        doc_id=_get_row_id(row,('id','doc_id','_id'))
        if doc_id is None:
            continue
        corpus_ids.append(doc_id)
        corpus_by_id[doc_id]=row

    sample_count=max(1,int(len(corpus_ids)*size_percent/100.0))
    rng=random.Random(seed)
    selected_doc_ids=set(rng.sample(corpus_ids,sample_count))

    sampled_corpus=[corpus_by_id[doc_id] for doc_id in corpus_ids if doc_id in selected_doc_ids]

    sampled_qrels={}
    for qid,doc_ids in qrels.items():
        kept={doc_id for doc_id in doc_ids if doc_id in selected_doc_ids}
        if kept:
            sampled_qrels[qid]=kept

    query_by_id={}
    for row in queries:
        qid=_get_row_id(row,('id','query_id','_id'))
        if qid is None:
            continue
        query_by_id[qid]=row

    sampled_queries=[query_by_id[qid] for qid in sampled_qrels.keys() if qid in query_by_id]

    meta={
        "size_percent":size_percent,
        "random_seed":seed,
        "original_num_docs":original_doc_count,
        "sampled_num_docs":len(sampled_corpus),
        "num_queries":len(sampled_queries),
        "num_qrels_queries":len(sampled_qrels)
    }

    return sampled_corpus,sampled_queries,sampled_qrels,meta

def load_dataset(config,dataset_name,verbose=True,size_percent=100.0,seed=DEFAULT_RANDOM_SEED):
    beir=config['beir_datasets'][dataset_name]
    dir_path=beir['dir']

    if verbose:
        print(f"[data] loading dataset={dataset_name}")
        print(f"[data] base={dir_path}")

    corpus_path=os.path.join(dir_path,'corpus.jsonl')
    queries_path=os.path.join(dir_path,'queries.jsonl')
    qrels_path=os.path.join(dir_path,'qrels','test.tsv')

    corpus=load_jsonl(corpus_path)
    queries=load_jsonl(queries_path)
    qrels=load_qrels(qrels_path)

    corpus,queries,qrels,meta=sample_corpus_and_align(corpus,queries,qrels,size_percent,seed)

    if verbose:
        print(f"[data] corpus loaded: {meta['sampled_num_docs']} rows")
        print(f"[data] queries loaded: {meta['num_queries']} rows")
        print(f"[data] qrels loaded: {meta['num_qrels_queries']} queries")
        if meta['size_percent']<100.0:
            print(f"[data] sampled size: {meta['size_percent']}% | seed={meta['random_seed']} | original_docs={meta['original_num_docs']}")

    return corpus,queries,qrels,meta
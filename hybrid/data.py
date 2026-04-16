import json
import os

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

def load_dataset(config,dataset_name):
    beir=config['beir_datasets'][dataset_name]
    dir_path=beir['dir']

    print(f"[data] loading dataset={dataset_name}")
    print(f"[data] base={dir_path}")

    corpus_path=os.path.join(dir_path,'corpus.jsonl')
    queries_path=os.path.join(dir_path,'queries.jsonl')
    qrels_path=os.path.join(dir_path,'qrels','test.tsv')

    corpus=load_jsonl(corpus_path)
    print(f"[data] corpus loaded: {len(corpus)} rows")

    queries=load_jsonl(queries_path)
    print(f"[data] queries loaded: {len(queries)} rows")

    qrels=load_qrels(qrels_path)
    print(f"[data] qrels loaded: {len(qrels)} queries")

    return corpus,queries,qrels
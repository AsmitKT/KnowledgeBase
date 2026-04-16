import json
import os

def load_jsonl(path):
    data=[]
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_dataset(config,dataset_name):
    beir=config['beir_datasets'][dataset_name]
    dir_path=beir['dir']
    corpus=load_jsonl(os.path.join(dir_path,'corpus.jsonl'))
    queries=load_jsonl(os.path.join(dir_path,'queries.jsonl'))
    qrels={}
    qrels_path=os.path.join(dir_path,'qrels','test.tsv')
    if os.path.exists(qrels_path):
        with open(qrels_path) as f:
            for line in f:
                parts=line.strip().split('\t')
                if len(parts)>=3:
                    qid=parts[0]
                    doc_id=parts[2]
                    qrels.setdefault(qid,set()).add(doc_id)
    return corpus,queries,qrels
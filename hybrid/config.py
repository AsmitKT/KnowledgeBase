import json
import os

HYBRID_PARAMS={"artifacts_root":"artifacts","bm25":{"k1":1.2,"b":0.75},"dense":{"model_name":"Alibaba-NLP/gte-modernbert-base","batch_size":16,"normalize":True},"ann":{"m":16,"ef_construction":200,"ef_search":100},"fusion":{"rrf_k":60,"bm25_weight":1.0,"dense_weight":1.0,"metadata_weight":0.3}}

def load_config(path='datasets_config.json'):
    with open(path) as f:
        cfg=json.load(f)
    cfg['hybrid']=HYBRID_PARAMS
    return cfg

def ensure_artifacts_dirs(config):
    root=config['hybrid']['artifacts_root']
    for sub in ['indexes','embeddings','runs','logs']:
        os.makedirs(os.path.join(root,sub),exist_ok=True)
    return root
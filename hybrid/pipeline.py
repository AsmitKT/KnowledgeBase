import os
import pickle
from .config import ensure_artifacts_dirs
from .data import load_dataset
from .bm25 import BM25
from .dense import DenseRetrieval
from .ann import GraphANN
from .metadata import MetadataScorer
from .fusion import rrf_fuse
from .metrics import compute_metrics
from .metadata import build_query_text

def build_indexes(config,dataset_name):
    print(f"[pipeline] build start | dataset={dataset_name}")

    corpus,_,_=load_dataset(config,dataset_name)
    ensure_artifacts_dirs(config)

    print("[pipeline] step 1/4 | BM25 build")
    bm=BM25(config['hybrid']['bm25']['k1'],config['hybrid']['bm25']['b'])
    bm.build(corpus)

    print("[pipeline] step 2/4 | dense build")
    dr=DenseRetrieval(
        config['hybrid']['dense']['model_name'],
        config['hybrid']['dense']['batch_size'],
        normalize=config['hybrid']['dense']['normalize']
    )
    dr.build(corpus)

    print("[pipeline] step 3/4 | ANN build")
    ann=GraphANN(
        m=config['hybrid']['ann']['m'],
        ef_construction=config['hybrid']['ann'].get('ef_construction',64),
        ef_search=config['hybrid']['ann'].get('ef_search',64),
        random_seed=config['globals'].get('seed',42)
    )
    ann.build(dr.doc_ids,dr.embeddings)

    print("[pipeline] step 4/4 | saving artifacts")
    art=config['hybrid']['artifacts_root']
    prefix=os.path.join(art,dataset_name)
    os.makedirs(prefix,exist_ok=True)

    with open(os.path.join(prefix,'bm25.pkl'),'wb') as f:
        pickle.dump(bm,f)
    print("[pipeline] saved bm25.pkl")

    with open(os.path.join(prefix,'dense.pkl'),'wb') as f:
        pickle.dump(dr,f)
    print("[pipeline] saved dense.pkl")

    with open(os.path.join(prefix,'ann.pkl'),'wb') as f:
        pickle.dump(ann,f)
    print("[pipeline] saved ann.pkl")

    with open(os.path.join(prefix,'corpus.pkl'),'wb') as f:
        pickle.dump(corpus,f)
    print("[pipeline] saved corpus.pkl")

    print(f"[pipeline] build complete | dataset={dataset_name}")

def load_indexes(config,dataset_name):
    art=config['hybrid']['artifacts_root']
    prefix=os.path.join(art,dataset_name)

    print(f"[pipeline] loading artifacts | dataset={dataset_name}")

    with open(os.path.join(prefix,'bm25.pkl'),'rb') as f:
        bm=pickle.load(f)
    print("[pipeline] loaded bm25.pkl")

    with open(os.path.join(prefix,'dense.pkl'),'rb') as f:
        dr=pickle.load(f)
    print("[pipeline] loaded dense.pkl")

    with open(os.path.join(prefix,'ann.pkl'),'rb') as f:
        ann=pickle.load(f)
    print("[pipeline] loaded ann.pkl")

    with open(os.path.join(prefix,'corpus.pkl'),'rb') as f:
        corpus=pickle.load(f)
    print("[pipeline] loaded corpus.pkl")

    return bm,dr,ann,corpus

def search_query(config,dataset_name,query_text,top_k):
    print(f"[pipeline] search start | dataset={dataset_name} | top_k={top_k}")

    bm,dr,ann,corpus=load_indexes(config,dataset_name)
    meta=MetadataScorer()

    print("[pipeline] step 1/5 | BM25 retrieval")
    bm_res=bm.retrieve(query_text,top_k*5)

    print("[pipeline] step 2/5 | dense retrieval")
    dense_res=dr.query(query_text,top_k*5)

    print("[pipeline] step 3/5 | seeded ANN retrieval")
    seeds=[doc_id for doc_id,_ in bm_res]
    q_emb=dr.encode_texts([query_text],prefix="ann-query")[0]
    ann_res=ann.search(q_emb,seeds,top_k*5)

    print("[pipeline] step 4/5 | metadata scoring")
    meta_scores={}
    for idx,doc in enumerate(corpus,1):
        raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
        if raw_doc_id is None:
            continue
        doc_id=str(raw_doc_id)
        meta_scores[doc_id]=meta.score(query_text,doc.get('metadata',{}))
        if idx % 500 == 0 or idx == len(corpus):
            print(f"[pipeline] metadata scored {idx}/{len(corpus)} documents")

    meta_list=sorted(meta_scores.items(),key=lambda x:x[1],reverse=True)[:top_k*5]

    print("[pipeline] step 5/5 | RRF fusion")
    lists={
        'bm25':bm_res,
        'dense':dense_res,
        'ann':ann_res,
        'meta':meta_list
    }

    weights={
        'bm25':config['hybrid']['fusion']['bm25_weight'],
        'dense':config['hybrid']['fusion']['dense_weight'],
        'ann':config['hybrid']['fusion']['dense_weight'],
        'meta':config['hybrid']['fusion']['metadata_weight']
    }

    fused=rrf_fuse(lists,weights,config['hybrid']['fusion']['rrf_k'],top_k)

    print(f"[pipeline] search complete | returned={len(fused)}")
    return fused

def evaluate(config,dataset_name,top_k):
    print(f"[pipeline] eval start | dataset={dataset_name} | top_k={top_k}")

    _,queries,qrels=load_dataset(config,dataset_name)
    run={}

    for idx,q in enumerate(queries,1):
        qid=str(q.get('id') or q.get('query_id') or q.get('_id'))
        query_text=build_query_text(q,include_metadata=True)
        print(f"[pipeline] eval query {idx}/{len(queries)} | qid={qid}")
        res=search_query(config,dataset_name,query_text,top_k)
        run[qid]=[doc_id for doc_id,_ in res]

    metrics=compute_metrics(run,qrels,top_k)
    print(f"[pipeline] eval complete | metrics={metrics}")
    return metrics
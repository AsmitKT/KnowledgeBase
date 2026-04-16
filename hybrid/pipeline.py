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

def build_indexes(config,dataset_name):
    corpus,_,_=load_dataset(config,dataset_name)
    ensure_artifacts_dirs(config)

    bm=BM25(config['hybrid']['bm25']['k1'],config['hybrid']['bm25']['b'])
    bm.build(corpus)

    dr=DenseRetrieval(
        config['hybrid']['dense']['model_name'],
        config['hybrid']['dense']['batch_size'],
        normalize=config['hybrid']['dense']['normalize']
    )
    dr.build(corpus)

    ann=GraphANN(config['hybrid']['ann']['m'])
    ann.build(dr.doc_ids,dr.embeddings)

    art=config['hybrid']['artifacts_root']
    prefix=os.path.join(art,dataset_name)
    os.makedirs(prefix,exist_ok=True)

    with open(os.path.join(prefix,'bm25.pkl'),'wb') as f:
        pickle.dump(bm,f)
    with open(os.path.join(prefix,'dense.pkl'),'wb') as f:
        pickle.dump(dr,f)
    with open(os.path.join(prefix,'ann.pkl'),'wb') as f:
        pickle.dump(ann,f)
    with open(os.path.join(prefix,'corpus.pkl'),'wb') as f:
        pickle.dump(corpus,f)

def load_indexes(config,dataset_name):
    art=config['hybrid']['artifacts_root']
    prefix=os.path.join(art,dataset_name)

    with open(os.path.join(prefix,'bm25.pkl'),'rb') as f:
        bm=pickle.load(f)
    with open(os.path.join(prefix,'dense.pkl'),'rb') as f:
        dr=pickle.load(f)
    with open(os.path.join(prefix,'ann.pkl'),'rb') as f:
        ann=pickle.load(f)
    with open(os.path.join(prefix,'corpus.pkl'),'rb') as f:
        corpus=pickle.load(f)

    return bm,dr,ann,corpus

def search_query(config,dataset_name,query_text,top_k):
    bm,dr,ann,corpus=load_indexes(config,dataset_name)
    meta=MetadataScorer()

    bm_res=bm.retrieve(query_text,top_k*5)
    dense_res=dr.query(query_text,top_k*5)

    seeds=[doc_id for doc_id,_ in bm_res]
    q_emb=dr.encode_texts([query_text])[0]
    ann_res=ann.search(q_emb,seeds,top_k*5)

    meta_scores={}
    for doc in corpus:
        raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
        if raw_doc_id is None:
            continue
        doc_id=str(raw_doc_id)
        meta_scores[doc_id]=meta.score(query_text,doc.get('metadata',{}))

    meta_list=sorted(meta_scores.items(),key=lambda x:x[1],reverse=True)[:top_k*5]

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
    return fused

def evaluate(config,dataset_name,top_k):
    corpus,queries,qrels=load_dataset(config,dataset_name)
    run={}

    for q in queries:
        qid=str(q.get('id') or q.get('query_id') or q.get('_id'))
        query_text=q.get('text') or q.get('question') or ""
        res=search_query(config,dataset_name,query_text,top_k)
        run[qid]=[doc_id for doc_id,_ in res]

    return compute_metrics(run,qrels,top_k)
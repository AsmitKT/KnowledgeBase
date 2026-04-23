import os
import pickle
from .config import ensure_artifacts_dirs
from .data import load_dataset
from .bm25 import BM25
from .dense import DenseRetrieval
from .ann import GraphANN
from .metadata import MetadataScorer, build_query_text
from .fusion import rrf_fuse
from .metrics import compute_metrics
from .progress import TerminalProgressBar

def _artifact_dataset_name(dataset_name,size_percent):
    if float(size_percent)>=100.0:
        return dataset_name
    size_str=f"{float(size_percent):.4f}".rstrip('0').rstrip('.')
    size_str=size_str.replace('.','p')
    return f"{dataset_name}__size_{size_str}"

def build_indexes(config,dataset_name,size_percent=100.0):
    bar=TerminalProgressBar(5,label=f"build {dataset_name}")
    bar.update(0,message="loading dataset")
    corpus,_,_,meta=load_dataset(
        config,
        dataset_name,
        verbose=False,
        size_percent=size_percent,
        seed=config['globals'].get('seed',42)
    )
    ensure_artifacts_dirs(config)

    bar.update(1,message="building BM25")
    bm=BM25(config['hybrid']['bm25']['k1'],config['hybrid']['bm25']['b'])
    bm.build(corpus)

    bar.update(2,message="building dense")
    dr=DenseRetrieval(
        config['hybrid']['dense']['model_name'],
        config['hybrid']['dense']['batch_size'],
        normalize=config['hybrid']['dense']['normalize']
    )
    dr.build(corpus)

    bar.update(3,message="building ANN")
    ann=GraphANN(
        m=config['hybrid']['ann']['m'],
        ef_construction=config['hybrid']['ann'].get('ef_construction',64),
        ef_search=config['hybrid']['ann'].get('ef_search',64),
        random_seed=config['globals'].get('seed',42)
    )
    ann.build(dr.doc_ids,dr.embeddings)

    bar.update(4,message="saving artifacts")
    art=config['hybrid']['artifacts_root']
    dataset_artifact_name=_artifact_dataset_name(dataset_name,size_percent)
    prefix=os.path.join(art,dataset_artifact_name)
    os.makedirs(prefix,exist_ok=True)

    with open(os.path.join(prefix,'bm25.pkl'),'wb') as f:
        pickle.dump(bm,f)

    with open(os.path.join(prefix,'dense.pkl'),'wb') as f:
        pickle.dump(dr,f)

    with open(os.path.join(prefix,'ann.pkl'),'wb') as f:
        pickle.dump(ann,f)

    with open(os.path.join(prefix,'corpus.pkl'),'wb') as f:
        pickle.dump(corpus,f)

    with open(os.path.join(prefix,'build_meta.pkl'),'wb') as f:
        pickle.dump(meta,f)

    bar.finish("complete")

def load_indexes(config,dataset_name,size_percent=100.0):
    art=config['hybrid']['artifacts_root']
    dataset_artifact_name=_artifact_dataset_name(dataset_name,size_percent)
    prefix=os.path.join(art,dataset_artifact_name)

    with open(os.path.join(prefix,'bm25.pkl'),'rb') as f:
        bm=pickle.load(f)
    with open(os.path.join(prefix,'dense.pkl'),'rb') as f:
        dr=pickle.load(f)
    with open(os.path.join(prefix,'ann.pkl'),'rb') as f:
        ann=pickle.load(f)
    with open(os.path.join(prefix,'corpus.pkl'),'rb') as f:
        corpus=pickle.load(f)

    return bm,dr,ann,corpus

def _prepare_query_text(query_text,query_metadata=None):
    if query_metadata:
        query_obj={
            "text":query_text,
            "metadata":query_metadata
        }
        return build_query_text(query_obj,include_metadata=True)
    return query_text

def _run_search_with_indexes(config,bm,dr,ann,corpus,query_text,top_k,progress=None):
    meta=MetadataScorer()

    if progress is not None:
        progress.update(1,message="BM25 retrieve")
    bm_res=bm.retrieve(query_text,top_k*5)

    if progress is not None:
        progress.update(2,message="dense retrieve")
    dense_res=dr.query(query_text,top_k*5)

    seeds=[doc_id for doc_id,_ in bm_res]
    q_emb=dr.encode_texts([query_text])[0]

    if progress is not None:
        progress.update(3,message="ANN retrieve")
    ann_res=ann.search(q_emb,seeds,top_k*5)

    meta_scores={}
    for doc in corpus:
        raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
        if raw_doc_id is None:
            continue
        doc_id=str(raw_doc_id)
        meta_scores[doc_id]=meta.score(query_text,doc.get('metadata',{}))

    if progress is not None:
        progress.update(4,message="metadata score")
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

    if progress is not None:
        progress.update(5,message="fusion")
    return rrf_fuse(lists,weights,config['hybrid']['fusion']['rrf_k'],top_k)

def search_query(config,dataset_name,query_text,top_k,query_metadata=None,size_percent=100.0):
    final_query_text=_prepare_query_text(query_text,query_metadata)
    bar=TerminalProgressBar(5,label=f"search {dataset_name}")
    bar.update(0,message="loading indexes")
    bm,dr,ann,corpus=load_indexes(config,dataset_name,size_percent=size_percent)
    results=_run_search_with_indexes(config,bm,dr,ann,corpus,final_query_text,top_k,progress=bar)
    bar.finish("complete")
    return results

def evaluate(config,dataset_name,top_k,size_percent=100.0):
    _,queries,qrels,_=load_dataset(
        config,
        dataset_name,
        verbose=False,
        size_percent=size_percent,
        seed=config['globals'].get('seed',42)
    )
    bm,dr,ann,corpus=load_indexes(config,dataset_name,size_percent=size_percent)

    run={}
    total_queries=len(queries)
    bar=TerminalProgressBar(total_queries,label=f"eval {dataset_name}")
    bar.update(0,message=f"query 0/{total_queries}")

    for idx,q in enumerate(queries,1):
        qid=str(q.get('id') or q.get('query_id') or q.get('_id'))
        query_text=build_query_text(q,include_metadata=True)
        res=_run_search_with_indexes(config,bm,dr,ann,corpus,query_text,top_k)
        run[qid]=[doc_id for doc_id,_ in res]
        bar.update(idx,message=f"query {idx}/{total_queries} | qid={qid}")

    metrics=compute_metrics(run,qrels,top_k)
    bar.finish("complete")
    return metrics
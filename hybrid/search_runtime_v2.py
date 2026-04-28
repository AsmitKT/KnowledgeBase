from .pipeline import load_indexes,evaluate_with_search_fn,prepare_query_text
from .metadata import MetadataScorer
from .progress import TerminalProgressBar

def _get_doc_id(doc):
    raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
    if raw_doc_id is None:
        return None
    return str(raw_doc_id)

def _build_doc_lookup(corpus):
    lookup={}
    for doc in corpus:
        doc_id=_get_doc_id(doc)
        if doc_id is not None:
            lookup[doc_id]=doc
    return lookup

def _build_dense_lookup(dr):
    return {str(doc_id):idx for idx,doc_id in enumerate(dr.doc_ids)}

def _metadata_prioritize_candidates(query_text,bm_res,corpus):
    if not bm_res:
        return []

    meta=MetadataScorer()
    doc_lookup=_build_doc_lookup(corpus)

    max_bm25=max(score for _,score in bm_res)
    if max_bm25<=0:
        max_bm25=1.0

    prioritized=[]
    for rank,(doc_id,bm25_score) in enumerate(bm_res):
        doc_id=str(doc_id)
        doc=doc_lookup.get(doc_id,{})
        metadata_score=meta.score(query_text,doc.get('metadata',{}))
        normalized_bm25=float(bm25_score)/float(max_bm25)
        prioritized.append((doc_id,float(bm25_score),normalized_bm25,float(metadata_score),rank))

    prioritized.sort(key=lambda x:(x[3]>0,x[3],x[2],-x[4]),reverse=True)
    return prioritized

def run_search_v2(config,bm,dr,ann,corpus,query_text,top_k,progress=None):
    bm25_pool=max(top_k*20,top_k)
    dense_pool=max(top_k*10,top_k)
    metadata_boost=0.05
    bm25_tie_boost=0.001

    if progress is not None:
        progress.update(1,message="BM25 candidate pool")
    bm_res=bm.retrieve(query_text,bm25_pool)

    if progress is not None:
        progress.update(2,message="metadata prioritize")
    prioritized=_metadata_prioritize_candidates(query_text,bm_res,corpus)

    selected=prioritized[:dense_pool]
    dense_lookup=_build_dense_lookup(dr)

    valid=[]
    for doc_id,bm25_score,normalized_bm25,metadata_score,rank in selected:
        idx=dense_lookup.get(doc_id)
        if idx is not None:
            valid.append((doc_id,idx,bm25_score,normalized_bm25,metadata_score,rank))

    if progress is not None:
        progress.update(3,message="dense rerank subset")

    if not valid:
        return []

    q_emb=dr.encode_texts([query_text])[0]
    scored=[]

    for doc_id,idx,bm25_score,normalized_bm25,metadata_score,rank in valid:
        dense_score=float((dr.embeddings[idx]@q_emb).item())
        final_score=dense_score+(metadata_boost*metadata_score)+(bm25_tie_boost*normalized_bm25)
        scored.append((doc_id,final_score))

    scored.sort(key=lambda x:x[1],reverse=True)

    if progress is not None:
        progress.update(4,message="complete")
    return scored[:top_k]

def search_query_v2(config,dataset_name,query_text,top_k,query_metadata=None,size_percent=100.0):
    final_query_text=prepare_query_text(query_text,query_metadata)
    bar=TerminalProgressBar(4,label=f"search {dataset_name} v2")
    bar.update(0,message="loading indexes")
    bm,dr,ann,corpus=load_indexes(config,dataset_name,size_percent=size_percent)
    res=run_search_v2(config,bm,dr,ann,corpus,final_query_text,top_k,progress=bar)
    bar.finish("complete")
    return res

def evaluate_v2(config,dataset_name,top_k,size_percent=100.0):
    return evaluate_with_search_fn(
        config,
        dataset_name,
        top_k,
        run_search_v2,
        size_percent=size_percent
    )
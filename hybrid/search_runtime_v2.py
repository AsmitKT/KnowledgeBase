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

def _normalize_dense_scores(scored_items):
    if not scored_items:
        return []

    values=[item[6] for item in scored_items]
    min_score=min(values)
    max_score=max(values)

    if max_score==min_score:
        return [(item[0],item[1],item[2],item[3],item[4],item[5],1.0) for item in scored_items]

    normalized=[]
    for doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,dense_score in scored_items:
        normalized_dense=(dense_score-min_score)/(max_score-min_score)
        normalized.append((doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,normalized_dense))
    return normalized

def _metadata_score_candidates(query_text,bm_res,corpus):
    if not bm_res:
        return []

    meta=MetadataScorer()
    doc_lookup=_build_doc_lookup(corpus)

    max_bm25=max(score for _,score in bm_res)
    if max_bm25<=0:
        max_bm25=1.0

    raw=[]
    max_metadata=0.0

    for rank,(doc_id,bm25_score) in enumerate(bm_res):
        doc_id=str(doc_id)
        doc=doc_lookup.get(doc_id,{})
        metadata_score=float(meta.score(query_text,doc.get('metadata',{})))
        if metadata_score>max_metadata:
            max_metadata=metadata_score
        normalized_bm25=float(bm25_score)/float(max_bm25)
        raw.append((doc_id,float(bm25_score),normalized_bm25,metadata_score,rank))

    if max_metadata<=0:
        max_metadata=1.0

    scored=[]
    for doc_id,bm25_score,normalized_bm25,metadata_score,rank in raw:
        normalized_metadata=metadata_score/max_metadata
        scored.append((doc_id,bm25_score,normalized_bm25,metadata_score,normalized_metadata,rank))

    scored.sort(key=lambda x:(x[4]>0,x[4],x[2],-x[5]),reverse=True)
    return scored

def run_search_v2(config,bm,dr,ann,corpus,query_text,top_k,progress=None):
    bm25_pool=max(top_k*20,top_k)
    dense_pool=max(top_k*10,top_k)

    bm25_weight=0.33
    dense_weight=0.65
    metadata_weight=0.02

    if progress is not None:
        progress.update(1,message="BM25 candidate pool")
    bm_res=bm.retrieve(query_text,bm25_pool)

    if progress is not None:
        progress.update(2,message="metadata candidate priority")
    candidates=_metadata_score_candidates(query_text,bm_res,corpus)

    selected=candidates[:dense_pool]
    dense_lookup=_build_dense_lookup(dr)

    valid=[]
    for doc_id,bm25_score,normalized_bm25,metadata_score,normalized_metadata,rank in selected:
        idx=dense_lookup.get(doc_id)
        if idx is not None:
            valid.append((doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,rank))

    if progress is not None:
        progress.update(3,message="dense rerank subset")

    if not valid:
        return []

    q_emb=dr.encode_texts([query_text])[0]
    dense_scored=[]

    for doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,rank in valid:
        dense_score=float((dr.embeddings[idx]@q_emb).item())
        dense_scored.append((doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,dense_score))

    dense_scored=_normalize_dense_scores(dense_scored)

    final=[]
    for doc_id,idx,bm25_score,normalized_bm25,metadata_score,normalized_metadata,normalized_dense in dense_scored:
        final_score=(bm25_weight*normalized_bm25)+(dense_weight*normalized_dense)+(metadata_weight*normalized_metadata)
        final.append((doc_id,final_score))

    final.sort(key=lambda x:x[1],reverse=True)

    if progress is not None:
        progress.update(4,message="weighted rerank")
    return final[:top_k]

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
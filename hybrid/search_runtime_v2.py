from .pipeline import load_indexes,evaluate_with_search_fn,_prepare_query_text
from .fusion import rrf_fuse
from .progress import TerminalProgressBar

def run_search_v2(config,bm,dr,ann,corpus,query_text,top_k,progress=None):
    bm_res=bm.retrieve(query_text,top_k*5)
    seeds=[doc_id for doc_id,_ in bm_res]
    q_emb=dr.encode_texts([query_text])[0]
    ann_res=ann.search(q_emb,seeds,top_k*5)

    lists={
        'bm25':bm_res,
        'ann':ann_res
    }

    weights={
        'bm25':config['hybrid']['fusion']['bm25_weight'],
        'ann':config['hybrid']['fusion']['dense_weight']
    }

    return rrf_fuse(lists,weights,config['hybrid']['fusion']['rrf_k'],top_k)

def search_query_v2(config,dataset_name,query_text,top_k,query_metadata=None,size_percent=100.0):
    final_query_text=_prepare_query_text(query_text,query_metadata)
    bar=TerminalProgressBar(3,label=f"search {dataset_name} v2")
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
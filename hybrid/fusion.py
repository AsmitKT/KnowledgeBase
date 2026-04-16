def rrf_fuse(lists,weights=None,rrf_k=60,top_k=10):
    scores={}
    for name,lst in lists.items():
        w=weights.get(name,1.0) if weights else 1.0
        for rank,item in enumerate(lst):
            doc_id=item[0]
            s=w/(rrf_k+rank)
            scores[doc_id]=scores.get(doc_id,0.0)+s
    items=list(scores.items())
    items.sort(key=lambda x:x[1],reverse=True)
    return items[:top_k]
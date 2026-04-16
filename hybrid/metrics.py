import math

def recall_at_k(run,qrels,k):
    total=0
    found=0
    for qid,rel_docs in qrels.items():
        total+=len(rel_docs)
        retrieved=run.get(qid,[])[:k]
        for doc_id in retrieved:
            if doc_id in rel_docs:
                found+=1
    return found/total if total else 0.0

def mrr_at_k(run,qrels,k):
    total=0
    sum_rr=0.0
    for qid,rel_docs in qrels.items():
        total+=1
        retrieved=run.get(qid,[])[:k]
        rr=0.0
        for i,doc_id in enumerate(retrieved):
            if doc_id in rel_docs:
                rr=1.0/(i+1)
                break
        sum_rr+=rr
    return sum_rr/total if total else 0.0

def ndcg_at_k(run,qrels,k):
    total=0
    sum_ndcg=0.0
    for qid,rel_docs in qrels.items():
        total+=1
        retrieved=run.get(qid,[])[:k]
        rel=[1 if doc in rel_docs else 0 for doc in retrieved]
        dcg=0.0
        for i,r in enumerate(rel):
            if r:
                dcg+=1.0/math.log2(i+2)
        idcg=0.0
        m=min(k,len(rel_docs))
        for i in range(m):
            idcg+=1.0/math.log2(i+2)
        sum_ndcg+=dcg/idcg if idcg>0 else 0.0
    return sum_ndcg/total if total else 0.0

def compute_metrics(run,qrels,k):
    return {'recall':recall_at_k(run,qrels,k),'mrr':mrr_at_k(run,qrels,k),'ndcg':ndcg_at_k(run,qrels,k)}
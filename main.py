import argparse
from hybrid.config import load_config
from hybrid.pipeline import build_indexes, search_query, evaluate
from hybrid.search_runtime_v2 import search_query_v2,evaluate_v2

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['build','search','eval'])
    p.add_argument('--dataset',required=True)
    p.add_argument('--query')
    p.add_argument('--search_type',choices=['original','v2'],default='original')
    p.add_argument('--top_k',type=int,default=10)
    p.add_argument('--size',type=float,default=100.0)
    p.add_argument('--location')
    p.add_argument('--date')
    p.add_argument('--date_type')
    p.add_argument('--source')
    a=p.parse_args()

    cfg=load_config()

    if a.mode=='build':
        build_indexes(cfg,a.dataset,size_percent=a.size)
    elif a.mode=='search':
        if not a.query:
            print('search mode requires --query')
            return

        query_metadata={}
        if a.location:
            query_metadata['location']=a.location
        if a.date:
            query_metadata['date']=a.date
        if a.date_type:
            query_metadata['date_type']=a.date_type
        if a.source:
            query_metadata['source']=a.source

        if a.search_type=='v2':
            res=search_query_v2(
                cfg,
                a.dataset,
                a.query,
                a.top_k,
                query_metadata=query_metadata if query_metadata else None,
                size_percent=a.size
            )
        else:
            res=search_query(
                cfg,
                a.dataset,
                a.query,
                a.top_k,
                query_metadata=query_metadata if query_metadata else None,
                size_percent=a.size
            )

        for doc_id,score in res:
            print(f"{doc_id}\t{score}")
    elif a.mode=='eval':
        if a.search_type=='v2':
            m=evaluate_v2(cfg,a.dataset,a.top_k,size_percent=a.size)
        else:
            m=evaluate(cfg,a.dataset,a.top_k,size_percent=a.size)
        for k,v in m.items():
            print(f"{k}\t{v}")

if __name__=='__main__':
    main()
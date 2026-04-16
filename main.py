import argparse
from hybrid.config import load_config
from hybrid.pipeline import build_indexes, search_query, evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode')
    p.add_argument('--dataset',required=True)
    p.add_argument('--query')
    p.add_argument('--split',default='test')
    p.add_argument('--top_k',type=int,default=10)
    a=p.parse_args()
    cfg=load_config()
    if a.mode=='build':
        build_indexes(cfg,a.dataset)
    elif a.mode=='search':
        if not a.query:
            return
        res=search_query(cfg,a.dataset,a.query,a.top_k)
        for doc_id,score in res:
            print(f"{doc_id}\t{score}")
    elif a.mode=='eval':
        m=evaluate(cfg,a.dataset,a.split,a.top_k)
        for k,v in m.items():
            print(f"{k}\t{v}")
    else:
        print('unknown mode')

if __name__=='__main__':
    main()
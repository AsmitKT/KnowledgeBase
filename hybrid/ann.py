import random

class GraphANN:
    def __init__(self,m=16):
        self.m=m
        self.doc_ids=None
        self.embeddings=None
        self.neighbors=None
    def build(self,doc_ids,embeddings):
        self.doc_ids=doc_ids
        self.embeddings=embeddings
        n=len(doc_ids)
        self.neighbors={i:[] for i in range(n)}
        for i in range(n):
            inds=list(range(n))
            inds.remove(i)
            random.shuffle(inds)
            self.neighbors[i]=inds[:self.m]
    def search(self,query_vector,seeds,top_k):
        visited=set()
        scores={}
        if seeds:
            seed_indices=[self.doc_ids.index(s) for s in seeds if s in self.doc_ids]
        else:
            seed_indices=[random.randrange(len(self.doc_ids))]
        frontier=list(seed_indices)
        for idx in seed_indices:
            visited.add(idx)
        while frontier and len(scores)<top_k*5:
            idx=frontier.pop()
            emb=self.embeddings[idx]
            sim=float((emb@query_vector).item())
            scores[idx]=sim
            for nb in self.neighbors[idx]:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append(nb)
        pairs=[(self.doc_ids[i],scores[i]) for i in scores]
        pairs.sort(key=lambda x:x[1],reverse=True)
        return pairs[:top_k]
import heapq
import math
import random
import torch

class GraphANN:
    def __init__(self,m=16,ef_construction=64,ef_search=64,random_seed=42):
        self.m=max(2,int(m))
        self.ef_construction=max(self.m,int(ef_construction))
        self.ef_search=max(self.m,int(ef_search))
        self.random=random.Random(random_seed)
        self.doc_ids=[]
        self.embeddings=None
        self.id_to_index={}
        self.layers={}
        self.levels={}
        self.entry_point=None
        self.max_level=-1

    def _normalize_embeddings(self,embeddings):
        if not isinstance(embeddings,torch.Tensor):
            embeddings=torch.tensor(embeddings,dtype=torch.float32)
        embeddings=embeddings.detach().cpu().float()
        norms=embeddings.norm(dim=1,keepdim=True).clamp_min(1e-12)
        return embeddings/norms

    def _similarity(self,query_vector,node_idx):
        return float(torch.dot(query_vector,self.embeddings[node_idx]).item())

    def _sample_level(self):
        level=0
        while self.random.random()<0.5:
            level+=1
        return level

    def _ensure_layer(self,level):
        if level not in self.layers:
            self.layers[level]={}

    def _greedy_search_layer(self,query_vector,entry_idx,level):
        current=entry_idx
        current_score=self._similarity(query_vector,current)

        changed=True
        while changed:
            changed=False
            for nb in self.layers.get(level,{}).get(current,[]):
                score=self._similarity(query_vector,nb)
                if score>current_score:
                    current=nb
                    current_score=score
                    changed=True

        return current

    def _search_layer(self,query_vector,entry_points,level,ef):
        visited=set()
        candidate_heap=[]
        result_heap=[]

        for idx in entry_points:
            if idx is None:
                continue
            if idx in visited:
                continue
            visited.add(idx)
            score=self._similarity(query_vector,idx)
            heapq.heappush(candidate_heap,(-score,idx))
            heapq.heappush(result_heap,(score,idx))

        while candidate_heap:
            current_score_neg,current_idx=heapq.heappop(candidate_heap)
            current_score=-current_score_neg

            if len(result_heap)>=ef and current_score<result_heap[0][0]:
                break

            for nb in self.layers.get(level,{}).get(current_idx,[]):
                if nb in visited:
                    continue

                visited.add(nb)
                score=self._similarity(query_vector,nb)

                if len(result_heap)<ef or score>result_heap[0][0]:
                    heapq.heappush(candidate_heap,(-score,nb))
                    heapq.heappush(result_heap,(score,nb))
                    if len(result_heap)>ef:
                        heapq.heappop(result_heap)

        result_heap.sort(key=lambda x:x[0],reverse=True)
        return [idx for _,idx in result_heap]

    def _select_neighbors(self,query_vector,candidates,m):
        scored=[]
        seen=set()

        for idx in candidates:
            if idx in seen:
                continue
            seen.add(idx)
            scored.append((self._similarity(query_vector,idx),idx))

        scored.sort(key=lambda x:x[0],reverse=True)
        return [idx for _,idx in scored[:m]]

    def _prune_neighbors_for_node(self,node_idx,level):
        neighbors=self.layers[level].get(node_idx,[])
        if len(neighbors)<=self.m:
            return

        base_vector=self.embeddings[node_idx]
        scored=[(float(torch.dot(base_vector,self.embeddings[nb]).item()),nb) for nb in neighbors]
        scored.sort(key=lambda x:x[0],reverse=True)
        self.layers[level][node_idx]=[nb for _,nb in scored[:self.m]]

    def _connect_bidirectional(self,node_idx,neighbors,level):
        self.layers[level].setdefault(node_idx,[])

        for nb in neighbors:
            if nb==node_idx:
                continue

            if nb not in self.layers[level][node_idx]:
                self.layers[level][node_idx].append(nb)

            self.layers[level].setdefault(nb,[])
            if node_idx not in self.layers[level][nb]:
                self.layers[level][nb].append(node_idx)

            self._prune_neighbors_for_node(nb,level)

        self._prune_neighbors_for_node(node_idx,level)

    def build(self,doc_ids,embeddings):
        print(f"[ann] building hierarchical graph | nodes={len(doc_ids)} | m={self.m} | ef_construction={self.ef_construction}")

        self.doc_ids=[str(x) for x in doc_ids]
        self.embeddings=self._normalize_embeddings(embeddings)
        self.id_to_index={doc_id:i for i,doc_id in enumerate(self.doc_ids)}
        self.layers={}
        self.levels={}
        self.entry_point=None
        self.max_level=-1

        for idx in range(len(self.doc_ids)):
            level=self._sample_level()
            self.levels[idx]=level

            for l in range(level+1):
                self._ensure_layer(l)
                self.layers[l].setdefault(idx,[])

            if self.entry_point is None:
                self.entry_point=idx
                self.max_level=level
            else:
                query_vector=self.embeddings[idx]
                entry=self.entry_point

                for l in range(self.max_level,level,-1):
                    entry=self._greedy_search_layer(query_vector,entry,l)

                top_level=min(level,self.max_level)
                for l in range(top_level,-1,-1):
                    candidates=self._search_layer(query_vector,[entry],l,self.ef_construction)
                    candidates=[c for c in candidates if c!=idx]
                    neighbors=self._select_neighbors(query_vector,candidates,self.m)
                    self._connect_bidirectional(idx,neighbors,l)

                    if candidates:
                        entry=candidates[0]

                if level>self.max_level:
                    for l in range(self.max_level+1,level+1):
                        self._ensure_layer(l)
                        self.layers[l].setdefault(idx,[])
                    self.entry_point=idx
                    self.max_level=level

            if (idx+1)%100==0 or (idx+1)==len(self.doc_ids):
                print(f"[ann] inserted {idx+1}/{len(self.doc_ids)} nodes | current_max_level={self.max_level}")

        print(f"[ann] build complete | layers={self.max_level+1} | entry_point={self.entry_point}")

    def search(self,query_vector,seeds,top_k):
        if self.embeddings is None or len(self.doc_ids)==0:
            return []

        print(f"[ann] searching | top_k={top_k} | ef_search={self.ef_search} | seeds={len(seeds) if seeds else 0}")

        if not isinstance(query_vector,torch.Tensor):
            query_vector=torch.tensor(query_vector,dtype=torch.float32)
        query_vector=query_vector.detach().cpu().float()
        query_vector=query_vector/query_vector.norm().clamp_min(1e-12)

        entry_points=[]

        if seeds:
            for doc_id in seeds:
                idx=self.id_to_index.get(str(doc_id))
                if idx is not None:
                    entry_points.append(idx)

        if not entry_points:
            entry=self.entry_point
            for l in range(self.max_level,0,-1):
                entry=self._greedy_search_layer(query_vector,entry,l)
            entry_points=[entry]
        else:
            if self.entry_point is not None:
                entry_points.append(self.entry_point)

        ordered=self._search_layer(query_vector,entry_points,0,max(top_k,self.ef_search))
        pairs=[(self.doc_ids[idx],self._similarity(query_vector,idx)) for idx in ordered]
        pairs.sort(key=lambda x:x[1],reverse=True)

        print(f"[ann] search complete | returned={min(top_k,len(pairs))}")
        return pairs[:top_k]
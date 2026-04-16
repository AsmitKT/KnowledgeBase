import math
import re

class BM25:
    def __init__(self,k1=1.2,b=0.75):
        self.k1=k1
        self.b=b
    def tokenize(self,text):
        return re.findall(r'\b\w+\b',text.lower())
    def build(self,corpus):
        self.N=len(corpus)
        self.doc_len={}
        self.avg_len=0.0
        self.df={}
        self.tf={}
        for doc in corpus:
            doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
            tokens=self.tokenize((doc.get('title','')+' '+doc.get('text','')).strip())
            self.doc_len[doc_id]=len(tokens)
            self.avg_len+=len(tokens)
            seen=set()
            for t in tokens:
                self.tf.setdefault(t,{})[doc_id]=self.tf[t].get(doc_id,0)+1
                if t not in seen:
                    self.df[t]=self.df.get(t,0)+1
                    seen.add(t)
        if self.N:
            self.avg_len/=self.N
        else:
            self.avg_len=1.0
    def idf(self,term):
        df=self.df.get(term,0)
        return math.log(1+(self.N-df+0.5)/(df+0.5))
    def score(self,query_terms,doc_id):
        s=0.0
        for t in query_terms:
            postings=self.tf.get(t)
            if not postings or doc_id not in postings:
                continue
            f=postings[doc_id]
            dl=self.doc_len[doc_id]
            idf=self.idf(t)
            s+=idf*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*dl/self.avg_len))
        return s
    def retrieve(self,query_text,top_k):
        terms=self.tokenize(query_text)
        candidate_docs=set()
        for t in terms:
            postings=self.tf.get(t)
            if postings:
                candidate_docs.update(postings.keys())
        scores=[]
        for doc_id in candidate_docs:
            s=self.score(terms,doc_id)
            scores.append((doc_id,s))
        scores.sort(key=lambda x:x[1],reverse=True)
        return scores[:top_k]
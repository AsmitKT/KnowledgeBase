import math
import re

class BM25:
    def __init__(self,k1=1.2,b=0.75):
        self.k1=k1
        self.b=b

    def tokenize(self,text):
        return re.findall(r'\b\w+\b',str(text or "").lower())

    def build(self,corpus):
        self.doc_len={}
        self.avg_len=0.0
        self.df={}
        self.tf={}
        valid_docs=0

        for doc in corpus:
            raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
            if raw_doc_id is None:
                continue
            doc_id=str(raw_doc_id)
            tokens=self.tokenize((str(doc.get('title',''))+' '+str(doc.get('text',''))).strip())
            self.doc_len[doc_id]=len(tokens)
            self.avg_len+=len(tokens)
            valid_docs+=1
            seen=set()

            for t in tokens:
                postings=self.tf.setdefault(t,{})
                postings[doc_id]=postings.get(doc_id,0)+1
                if t not in seen:
                    self.df[t]=self.df.get(t,0)+1
                    seen.add(t)

        self.N=valid_docs
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
import torch
from transformers import AutoTokenizer,AutoModel

class DenseRetrieval:
    def __init__(self,model_name,batch_size=16,device=None,normalize=True):
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True)
        self.model=AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.batch_size=batch_size
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.normalize=normalize
        self.embeddings=None
        self.doc_ids=None
    def encode_texts(self,texts):
        all_emb=[]
        for i in range(0,len(texts),self.batch_size):
            batch=texts[i:i+self.batch_size]
            enc=self.tokenizer(batch,return_tensors='pt',padding=True,truncation=True)
            for k in enc:
                enc[k]=enc[k].to(self.device)
            with torch.no_grad():
                out=self.model(**enc)
            emb=out.last_hidden_state[:,0,:].cpu()
            if self.normalize:
                emb=emb/emb.norm(dim=1,keepdim=True)
            all_emb.append(emb)
        return torch.cat(all_emb,dim=0)
    def build(self,corpus):
        texts=[]
        ids=[]
        for doc in corpus:
            ids.append(doc.get('id') or doc.get('doc_id') or doc.get('_id'))
            texts.append(((doc.get('title','')+' '+doc.get('text','')).strip()))
        self.doc_ids=ids
        self.embeddings=self.encode_texts(texts)
    def query(self,query_text,top_k):
        q_emb=self.encode_texts([query_text])[0]
        sims=(self.embeddings@q_emb).tolist()
        pairs=list(zip(self.doc_ids,sims))
        pairs.sort(key=lambda x:x[1],reverse=True)
        return pairs[:top_k]
import torch
from transformers import AutoTokenizer,AutoModel
from .metadata import build_document_text

class DenseRetrieval:
    def __init__(self,model_name,batch_size=16,device=None,normalize=True):
        self.model_name=model_name
        self.batch_size=batch_size
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize=normalize

        print(f"[dense] loading tokenizer | model={self.model_name}")
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,use_fast=True)

        print(f"[dense] loading model | device={self.device}")
        self.model=AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        self.embeddings=None
        self.doc_ids=None

    def encode_texts(self,texts,prefix="dense"):
        total=len(texts)
        print(f"[{prefix}] encoding {total} texts | batch_size={self.batch_size}")

        all_emb=[]
        total_batches=(total+self.batch_size-1)//self.batch_size

        for batch_idx,i in enumerate(range(0,total,self.batch_size),1):
            batch=texts[i:i+self.batch_size]
            enc=self.tokenizer(batch,return_tensors='pt',padding=True,truncation=True)

            for k in enc:
                enc[k]=enc[k].to(self.device)

            with torch.no_grad():
                out=self.model(**enc)

            emb=out.last_hidden_state[:,0,:].cpu()

            if self.normalize:
                emb=emb/emb.norm(dim=1,keepdim=True).clamp_min(1e-12)

            all_emb.append(emb)
            print(f"[{prefix}] batch {batch_idx}/{total_batches} complete")

        combined=torch.cat(all_emb,dim=0)
        print(f"[{prefix}] encoding complete | shape={tuple(combined.shape)}")
        return combined

    def build(self,corpus):
        print(f"[dense] building document embeddings for {len(corpus)} documents")

        texts=[]
        ids=[]

        for doc in corpus:
            raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
            if raw_doc_id is None:
                continue
            ids.append(str(raw_doc_id))
            texts.append(build_document_text(doc,include_metadata=True))

        self.doc_ids=ids
        self.embeddings=self.encode_texts(texts,prefix="dense-build")

        print(f"[dense] build complete | docs={len(self.doc_ids)}")

    def query(self,query_text,top_k):
        print(f"[dense] retrieving | top_k={top_k}")

        q_emb=self.encode_texts([query_text],prefix="dense-query")[0]
        sims=(self.embeddings@q_emb).tolist()
        pairs=list(zip(self.doc_ids,sims))
        pairs.sort(key=lambda x:x[1],reverse=True)

        print(f"[dense] retrieval complete | returned={min(top_k,len(pairs))}")
        return pairs[:top_k]
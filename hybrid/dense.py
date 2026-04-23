import os
import torch
from transformers import AutoTokenizer,AutoModel
from .metadata import build_document_text

def _disable_external_progress():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")

    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers.utils import logging as transformers_logging
        fn=getattr(transformers_logging,"disable_progress_bar",None)
        if callable(fn):
            fn()
    except Exception:
        pass

class DenseRetrieval:
    def __init__(self,model_name,batch_size=16,device=None,normalize=True):
        self.model_name=model_name
        self.batch_size=batch_size
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalize=normalize

        _disable_external_progress()

        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,use_fast=True)
        self.model=AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        self.embeddings=None
        self.doc_ids=None

    def encode_texts(self,texts,prefix="dense",progress_callback=None):
        if not texts:
            return torch.empty((0,0),dtype=torch.float32)

        all_emb=[]
        total=len(texts)
        total_batches=(total+self.batch_size-1)//self.batch_size

        for batch_idx,i in enumerate(range(0,total,self.batch_size),start=1):
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

            if progress_callback is not None:
                progress_callback(batch_idx,total_batches,prefix)

        return torch.cat(all_emb,dim=0)

    def build(self,corpus,progress_callback=None):
        texts=[]
        ids=[]

        for doc in corpus:
            raw_doc_id=doc.get('id') or doc.get('doc_id') or doc.get('_id')
            if raw_doc_id is None:
                continue
            ids.append(str(raw_doc_id))
            texts.append(build_document_text(doc,include_metadata=True))

        self.doc_ids=ids
        self.embeddings=self.encode_texts(texts,prefix="dense-build",progress_callback=progress_callback)

    def query(self,query_text,top_k):
        q_emb=self.encode_texts([query_text],prefix="dense-query")[0]
        sims=(self.embeddings@q_emb).tolist()
        pairs=list(zip(self.doc_ids,sims))
        pairs.sort(key=lambda x:x[1],reverse=True)
        return pairs[:top_k]
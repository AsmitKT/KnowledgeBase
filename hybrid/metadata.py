import re

class MetadataScorer:
    def __init__(self):
        pass
    def tokenize(self,text):
        return re.findall(r'\b\w+\b',str(text).lower())
    def score(self,query,metadata):
        q=set(self.tokenize(query))
        s=0.0
        for v in metadata.values():
            if isinstance(v,str):
                if q.intersection(self.tokenize(v)):
                    s+=1.0
            elif isinstance(v,list):
                for val in v:
                    if q.intersection(self.tokenize(val)):
                        s+=1.0
        return s
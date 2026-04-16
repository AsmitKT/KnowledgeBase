import json
import re

def normalize_text(value):
    return " ".join(str(value or "").strip().split())

def _flatten_metadata_value(value,prefix=""):
    parts=[]

    if value is None:
        return parts

    if isinstance(value,dict):
        for key,subvalue in value.items():
            key_str=normalize_text(key)
            next_prefix=f"{prefix}.{key_str}" if prefix else key_str
            parts.extend(_flatten_metadata_value(subvalue,next_prefix))
        return parts

    if isinstance(value,(list,tuple,set)):
        for item in value:
            parts.extend(_flatten_metadata_value(item,prefix))
        return parts

    value_str=normalize_text(value)
    if not value_str:
        return parts

    if prefix:
        parts.append(f"{prefix}: {value_str}")
    else:
        parts.append(value_str)

    return parts

def metadata_to_text(metadata):
    if metadata is None:
        return ""

    if isinstance(metadata,str):
        metadata=metadata.strip()
        if not metadata:
            return ""
        try:
            metadata=json.loads(metadata)
        except Exception:
            return normalize_text(metadata)

    parts=_flatten_metadata_value(metadata)
    seen=set()
    unique_parts=[]
    for part in parts:
        if part not in seen:
            seen.add(part)
            unique_parts.append(part)

    return " | ".join(unique_parts).strip()

def build_query_text(query,include_metadata=True):
    text=normalize_text(query.get("text") or query.get("question") or "")
    metadata=query.get("metadata",{})
    metadata_text=metadata_to_text(metadata) if include_metadata else ""

    parts=[]

    if isinstance(metadata,dict):
        location=normalize_text(metadata.get("location",""))
        date=normalize_text(metadata.get("date",""))
        date_type=normalize_text(metadata.get("date_type",""))

        if location:
            parts.append(f"location: {location}")
        if date:
            parts.append(f"date: {date}")
        if date_type:
            parts.append(f"date_type: {date_type}")

    if text:
        parts.append(f"question: {text}")

    if metadata_text:
        parts.append(f"metadata: {metadata_text}")

    return "\n".join(parts).strip()

def build_document_text(doc,include_metadata=True):
    title=normalize_text(doc.get("title",""))
    text=normalize_text(doc.get("text",""))
    metadata_text=metadata_to_text(doc.get("metadata",{})) if include_metadata else ""

    parts=[]
    if title:
        parts.append(f"title: {title}")
    if metadata_text:
        parts.append(f"metadata: {metadata_text}")
    if text:
        parts.append(f"text: {text}")

    return "\n".join(parts).strip()

class MetadataScorer:
    def __init__(self):
        pass

    def tokenize(self,text):
        return re.findall(r'\b\w+\b',str(text or "").lower())

    def score(self,query,metadata):
        q=set(self.tokenize(query))
        s=0.0

        for v in metadata.values():
            if isinstance(v,str):
                if q.intersection(self.tokenize(v)):
                    s+=1.0
            elif isinstance(v,(list,tuple,set)):
                for val in v:
                    if q.intersection(self.tokenize(val)):
                        s+=1.0
            elif isinstance(v,dict):
                meta_text=metadata_to_text(v)
                if q.intersection(self.tokenize(meta_text)):
                    s+=1.0

        return s
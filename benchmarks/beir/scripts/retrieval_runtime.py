import gc
import heapq

import numpy as np

def update_heap(heap, doc_id, score, top_k):
    item = (float(score), doc_id)
    if len(heap) < top_k:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)

def build_results_from_heaps(heaps):
    results = {}
    for qid, heap in heaps.items():
        ranked = sorted(heap, key=lambda x: x[0], reverse=True)
        results[qid] = {doc_id: score for score, doc_id in ranked}
    return results

def is_memory_error(exc):
    if isinstance(exc, MemoryError):
        return True
    name = exc.__class__.__name__
    if "MemoryError" in name or "ArrayMemoryError" in name:
        return True
    msg = str(exc)
    if "Unable to allocate" in msg or "out of memory" in msg.lower():
        return True
    return False

def encode_queries_once(queries, encode_query_fn):
    encoded = {}
    for qid, query_text in queries.items():
        encoded[qid] = encode_query_fn(query_text)
    return encoded

def run_full_search(corpus, queries, top_k, encode_corpus_fn, encode_query_fn, score_chunk_fn, exclude_self=False):
    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    documents = [corpus[doc_id] for doc_id in doc_ids]
    corpus_repr = encode_corpus_fn(documents)
    encoded_queries = encode_queries_once(queries, encode_query_fn)

    results = {}
    for qid in query_ids:
        query_repr = encoded_queries[qid]
        scores = score_chunk_fn(corpus_repr, query_repr)
        local_top_k = min(top_k, len(doc_ids))
        top_k_ind = np.argpartition(scores, -local_top_k)[-local_top_k:]
        top_k_ind = top_k_ind[np.argsort(scores[top_k_ind])[::-1]]
        results[qid] = {
            doc_ids[int(pid)]: float(scores[int(pid)])
            for pid in top_k_ind
            if not exclude_self or doc_ids[int(pid)] != qid
        }

    del corpus_repr
    gc.collect()
    return results

def run_chunked_search(corpus, queries, top_k, chunk_docs, encode_corpus_fn, encode_query_fn, score_chunk_fn, exclude_self=False):
    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    encoded_queries = encode_queries_once(queries, encode_query_fn)
    heaps = {qid: [] for qid in query_ids}

    chunk_docs = max(chunk_docs, top_k)
    total_docs = len(doc_ids)
    total_chunks = (total_docs + chunk_docs - 1) // chunk_docs

    for chunk_idx, start in enumerate(range(0, total_docs, chunk_docs), start=1):
        end = min(start + chunk_docs, total_docs)
        chunk_doc_ids = doc_ids[start:end]
        chunk_documents = [corpus[doc_id] for doc_id in chunk_doc_ids]

        print(f"Chunk {chunk_idx}/{total_chunks} | docs {start}:{end}")

        corpus_repr = encode_corpus_fn(chunk_documents)
        local_top_k = min(top_k, len(chunk_doc_ids))

        for qid in query_ids:
            query_repr = encoded_queries[qid]
            scores = score_chunk_fn(corpus_repr, query_repr)
            top_k_ind = np.argpartition(scores, -local_top_k)[-local_top_k:]
            top_k_ind = top_k_ind[np.argsort(scores[top_k_ind])[::-1]]

            heap = heaps[qid]
            for pid in top_k_ind:
                doc_id = chunk_doc_ids[int(pid)]
                if exclude_self and doc_id == qid:
                    continue
                update_heap(heap, doc_id, float(scores[int(pid)]), top_k)

        del corpus_repr
        gc.collect()

    return build_results_from_heaps(heaps)

def run_full_or_chunked_search(corpus, queries, top_k, chunk_docs, encode_corpus_fn, encode_query_fn, score_chunk_fn, exclude_self=False):
    total_docs = len(corpus)

    if chunk_docs > 0 and total_docs > chunk_docs:
        results = run_chunked_search(
            corpus,
            queries,
            top_k,
            chunk_docs,
            encode_corpus_fn,
            encode_query_fn,
            score_chunk_fn,
            exclude_self=exclude_self,
        )
        return results, "chunked", chunk_docs

    try:
        results = run_full_search(
            corpus,
            queries,
            top_k,
            encode_corpus_fn,
            encode_query_fn,
            score_chunk_fn,
            exclude_self=exclude_self,
        )
        return results, "full", None
    except Exception as exc:
        if not is_memory_error(exc):
            raise
        if chunk_docs <= 0:
            raise
        print("Memory pressure detected. Falling back to chunked retrieval.")
        gc.collect()
        results = run_chunked_search(
            corpus,
            queries,
            top_k,
            chunk_docs,
            encode_corpus_fn,
            encode_query_fn,
            score_chunk_fn,
            exclude_self=exclude_self,
        )
        return results, "chunked_fallback", chunk_docs
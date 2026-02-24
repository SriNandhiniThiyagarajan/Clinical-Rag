from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from bm25_store import BM25Store

COLLECTION_NAME = "clinical_chunks"

_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_client = QdrantClient(host="localhost", port=6333)

_bm25 = BM25Store()
_bm25.load()  # loads data/bm25.pkl


def _norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    if mx <= 0:
        return [0.0 for _ in scores]
    return [s / mx for s in scores]


def retrieve_hybrid(question: str, k_vec: int = 10, k_bm25: int = 10, k_final: int = 5) -> List[Dict[str, Any]]:
    # --- Vector search ---
    qvec = _model.encode(question).tolist()
    vec_hits = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=k_vec,
    ).points

    vec_items = []
    vec_scores = []
    for h in vec_hits:
        p = h.payload or {}
        vec_items.append({
            "score_vec": float(h.score),
            "doc_id": p.get("doc_id"),
            "title": p.get("title"),
            "page": p.get("page"),
            "chunk_id": p.get("chunk_id"),
            "text": p.get("text", ""),
        })
        vec_scores.append(float(h.score))

    vec_scores_n = _norm(vec_scores)

    # --- BM25 search ---
    bm_hits = _bm25.query(question, k=k_bm25)
    bm_scores = [h["bm25_score"] for h in bm_hits]
    bm_scores_n = _norm(bm_scores)

    bm_items = []
    for i, h in enumerate(bm_hits):
        bm_items.append({
            "score_bm25": float(bm_scores_n[i]),
            "doc_id": h.get("doc_id"),
            "title": h.get("doc_id"),
            "page": h.get("page"),
            "chunk_id": h.get("chunk_id"),
            "text": h.get("text", ""),
        })

    # --- Merge / dedupe by chunk_id ---
    merged: Dict[str, Dict[str, Any]] = {}

    # Add vector items
    for i, item in enumerate(vec_items):
        cid = item.get("chunk_id") or f"vec_{i}"
        merged[cid] = {
            **item,
            "vec_n": float(vec_scores_n[i]) if i < len(vec_scores_n) else 0.0,
            "bm25_n": 0.0,
        }

    # Add bm25 items
    for item in bm_items:
        cid = item.get("chunk_id")
        if not cid:
            continue
        if cid not in merged:
            merged[cid] = {
                **item,
                "vec_n": 0.0,
                "bm25_n": float(item.get("score_bm25", 0.0)),
            }
        else:
            merged[cid]["bm25_n"] = float(item.get("score_bm25", 0.0))

    # --- Final hybrid score ---
    # alpha controls semantic vs keyword balance
    alpha = 0.6
    results = []
    for cid, item in merged.items():
        hybrid_score = alpha * item.get("vec_n", 0.0) + (1 - alpha) * item.get("bm25_n", 0.0)
        results.append({
            "score": float(hybrid_score),
            "doc_id": item.get("doc_id"),
            "title": item.get("title"),
            "page": item.get("page"),
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text", ""),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k_final]
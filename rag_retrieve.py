from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "clinical_chunks"

_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_client = QdrantClient(host="localhost", port=6333)

def retrieve(question: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Returns top-k chunks from Qdrant with metadata for citations.
    """
    qvec = _model.encode(question).tolist()

    hits = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=k,
    ).points

    results = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "score": float(h.score),
            "doc_id": p.get("doc_id"),
            "title": p.get("title"),
            "page": p.get("page"),
            "chunk_id": p.get("chunk_id"),
            "text": p.get("text", ""),
        })
    return results
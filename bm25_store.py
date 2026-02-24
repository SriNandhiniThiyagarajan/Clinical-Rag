import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

BM25_PATH = Path("data/bm25.pkl")

def _tok(s: str) -> list[str]:
    return s.lower().split()

class BM25Store:
    def __init__(self):
        self.items = []  # each: {"text":..., "doc_id":..., "page":..., "chunk_id":...}
        self.bm25 = None

    def build(self, items: list[dict]):
        self.items = items
        corpus = [_tok(x["text"]) for x in items]
        self.bm25 = BM25Okapi(corpus)

    def query(self, q: str, k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(_tok(q))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in top_idx:
            out.append({**self.items[i], "bm25_score": float(scores[i])})
        return out

    def save(self):
        BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BM25_PATH, "wb") as f:
            pickle.dump({"items": self.items, "bm25": self.bm25}, f)

    def load(self):
        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.items = data["items"]
        self.bm25 = data["bm25"]
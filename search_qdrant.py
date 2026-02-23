from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "clinical_chunks"

def main():
    # 1) Load embedding model (same one used during ingestion)
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 2) Connect to Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # 3) Ask a test question (use radar-topic now because your PDF is radar-related)
    question = "What is Moving-Target Indication (MTI) and how does Doppler help?"
    qvec = model.encode(question).tolist()

    # 4) Search Qdrant
    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=qvec,
        limit=5,
    ).points

    # 5) Print results
    print("\nQUESTION:", question)
    print("-" * 60)

    for rank, h in enumerate(hits, start=1):
        p = h.payload
        text = p.get("text", "")
        preview = (text[:250] + "...") if len(text) > 250 else text

        print(f"\n#{rank}  score={h.score:.4f}")
        print(f"doc_id: {p.get('doc_id')} | page: {p.get('page')} | chunk_id: {p.get('chunk_id')}")
        print("preview:", preview)

if __name__ == "__main__":
    main()
from pathlib import Path
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

PDF_PATH = Path("data/raw_pdfs/sample.pdf")
COLLECTION_NAME = "clinical_chunks"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks

def main():
    if not PDF_PATH.exists():
        print(f"❌ PDF not found at: {PDF_PATH.resolve()}")
        print("✅ Put a PDF there and rename it to sample.pdf")
        return

    # 1) Load embedding model
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    vector_size = model.get_sentence_embedding_dimension()

    # 2) Connect Qdrant
    client = QdrantClient(host="localhost", port=6333)

    # 3) Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )
        print(f"✅ Created Qdrant collection: {COLLECTION_NAME}")

    # 4) Read PDF pages
    doc = fitz.open(str(PDF_PATH))
    points = []
    point_id = 1

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        chunks = chunk_text(text)

        # 5) Embed + prepare points
        vectors = model.encode(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors), start=1):
            payload = {
                "doc_id": PDF_PATH.name,
                "title": PDF_PATH.stem,
                "source": "local_pdf",
                "year": None,
                "page": page_index,
                "chunk_id": f"{PDF_PATH.stem}_p{page_index}_c{i}",
                "text": chunk,
            }

            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )
            point_id += 1

    # 6) Upload to Qdrant
    if not points:
        print("❌ No text chunks found in the PDF.")
        return

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded {len(points)} chunks to Qdrant from {PDF_PATH.name}")

if __name__ == "__main__":
    main()
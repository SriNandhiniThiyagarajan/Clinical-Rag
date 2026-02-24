from pathlib import Path
import fitz
from bm25_store import BM25Store

PDF_PATH = Path("data/raw_pdfs/sample.pdf")

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
        print("❌ sample.pdf not found")
        return

    doc = fitz.open(str(PDF_PATH))
    items = []

    for page_index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text or not text.strip():
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks, start=1):
            items.append({
                "doc_id": PDF_PATH.name,
                "page": page_index,
                "chunk_id": f"{PDF_PATH.stem}_p{page_index}_c{i}",
                "text": chunk,
            })

    store = BM25Store()
    store.build(items)
    store.save()
    print(f"✅ BM25 built with {len(items)} chunks and saved to data/bm25.pkl")

if __name__ == "__main__":
    main()
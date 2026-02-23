from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional
from rag_retrieve import retrieve

app = FastAPI()

#req structure
class QueryRequest(BaseModel):
    question:str 
    patient_age: Optional[int] = None
    symptoms :Optional[List[str]] =[]
    current_medication :Optional[List[str]] =[]

#res structure
class QueryResponse(BaseModel):
    summary:str
    recommendation:str
    evidence_level:str
    confidence_score: float
    citations: List[str]

@app.get("/")
def root():
    return {"message":"Clinical rag project is running"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    hits = retrieve(req.question, k=5)

    # Simple confidence rule (MVP)
    if not hits or hits[0]["score"] < 0.20:
        return QueryResponse(
            summary="Insufficient evidence found in the current database.",
            recommendation="Please add more guideline PDFs or refine the question.",
            evidence_level="N/A",
            confidence_score=0.0,
            citations=[]
        )

    # Build citations (simple format)
    citations = []
    for i, h in enumerate(hits, start=1):
        citations.append(
            f"[C{i}] {h['doc_id']} page {h['page']} (chunk={h['chunk_id']}) score={h['score']:.3f}"
        )

    # For now: return evidence-only response (no LLM yet)
    top = hits[0]["text"][:350].replace("\n", " ")

    return QueryResponse(
        summary="Top retrieved evidence snippet (MVP).",
        recommendation=top,
        evidence_level="Retrieved evidence",
        confidence_score=float(hits[0]["score"]),
        citations=citations
    )
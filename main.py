from fastapi import FastAPI
from pydantic import BaseModel
from typing import List,Optional

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


@app.post("/query",response_model=QueryResponse)
def query(req:QueryRequest):

        return QueryResponse(
        summary="This is a sample evidence-based summary.",
        recommendation="Start standard treatment as per guidelines.",
        evidence_level="Moderate",
        confidence_score=0.82,
        citations=["WHO-2022-Page-14", "PubMed-Study-2021"]
    )
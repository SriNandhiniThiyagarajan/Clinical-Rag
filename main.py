from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import re

from hybrid_retrieve import retrieve_hybrid
from ollama_llm import ollama_chat_json

app = FastAPI()


# ---------- Request / Response models ----------
class QueryRequest(BaseModel):
    question: str
    patient_age: Optional[int] = None
    symptoms: List[str] = []
    current_medication: List[str] = []


class QueryResponse(BaseModel):
    summary: str
    recommendation: str
    evidence_level: str
    confidence_score: float
    citations: List[str]


@app.get("/")
def root():
    return {"message": "Clinical rag project is running"}


# ---------- Citation helpers ----------
def only_uses_allowed_citations(text: str, max_k: int) -> bool:
    """
    Returns True only if:
    - There is at least one [C#] token
    - All tokens are within 1..max_k
    """
    found = re.findall(r"\[C(\d+)\]", text)
    if not found:
        return False
    nums = [int(x) for x in found]
    return all(1 <= n <= max_k for n in nums)


def json_cites_valid(obj: dict, max_k: int = 5) -> bool:
    """
    Validate:
    1) quotes[*].cite is "C1".."C5"
    2) summary + recommendation contain ONLY [C1]..[C5] (and at least one)
    """
    try:
        quotes = obj.get("quotes", [])
        if not isinstance(quotes, list):
            return False

        for q in quotes:
            if not isinstance(q, dict):
                return False
            cite = str(q.get("cite", "")).strip()
            cite = cite.replace("[", "").replace("]", "")  # allow "[C1]" or "C1"
            if not re.fullmatch(rf"C[1-{max_k}]", cite):
                return False
        summary = str(obj.get("summary", ""))
        recommendation = str(obj.get("recommendation", ""))
        combined = summary + " " + recommendation

        return only_uses_allowed_citations(combined, max_k=max_k)
    except Exception:
        return False


# ---------- Main endpoint ----------
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 1) Hybrid retrieve
    hits = retrieve_hybrid(req.question, k_vec=10, k_bm25=10, k_final=5)

    # 2) Safety fallback
    if not hits or hits[0]["score"] < 0.20:
        return QueryResponse(
            summary="Insufficient evidence found in the current database.",
            recommendation="Please add more guideline PDFs or refine the question.",
            evidence_level="N/A",
            confidence_score=0.0,
            citations=[],
        )

    # 3) Build citations + evidence pack
    citations: List[str] = []
    evidence_lines: List[str] = []

    for i, h in enumerate(hits, start=1):
        citations.append(f"[C{i}] {h['doc_id']} page {h['page']} (chunk={h['chunk_id']})")
        snippet = (h.get("text", "") or "").replace("\n", " ")
        snippet = snippet[:900]
        evidence_lines.append(f"[C{i}] {snippet}")

    # 4) Strict JSON-only prompt (with explicit bracket rules + example)
    system = (
        "You are a clinical decision support assistant.\n"
        "Use ONLY the EVIDENCE provided.\n"
        "Do NOT invent sources, URLs, DOIs, journals, authors, or references.\n"
        "Do NOT expand acronyms unless the evidence explicitly defines the full form.\n"
        "\n"
        "CITATION RULES (VERY IMPORTANT):\n"
        "- In 'summary' and 'recommendation', citations MUST be in square brackets like [C1].\n"
        "- Do NOT write C1 without brackets.\n"
        "- End every factual sentence with a bracket citation like ... [C2]\n"
        "- You may ONLY use [C1]..[C5].\n"
        "\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "quotes": [{"text": "...", "cite": "C1"}],\n'
        '  "summary": "...",\n'
        '  "recommendation": "...",\n'
        '  "evidence_level": "Low|Moderate|High"\n'
        "}\n"
        "\n"
        "QUOTE RULES:\n"
        "- Provide 2 quotes maximum.\n"
        "- Each quote must be copied from EVIDENCE.\n"
        "- Each quote must have cite exactly: C1..C5 (NO brackets in cite field).\n"
        "\n"
        "EXAMPLE JSON (follow this style):\n"
        '{"quotes":[{"text":"Example quote.", "cite":"C1"}],'
        '"summary":"Sentence. [C1]","recommendation":"Sentence. [C2]",'
        '"evidence_level":"Moderate"}\n'
    )

    user = (
        f"QUESTION:\n{req.question}\n\n"
        "EVIDENCE:\n" + "\n\n".join(evidence_lines)
    )

    # 5) Call Ollama in JSON mode
    try:
        answer_obj = ollama_chat_json(system=system, user=user)
        print("OLLAMA_JSON:", answer_obj)
    except Exception:
        return QueryResponse(
            summary="Insufficient evidence found in the current database.",
            recommendation="Failed to call Ollama or parse JSON output. Ensure Ollama is running.",
            evidence_level="N/A",
            confidence_score=float(hits[0]["score"]),
            citations=citations,
        )

    if not json_cites_valid(answer_obj, max_k=5):
        system_retry = system + (
            "\nCRITICAL: Your JSON is INVALID unless BOTH 'summary' and 'recommendation' "
            "contain at least one bracket citation like [C1]. Fix it now."
        )
        answer_obj = ollama_chat_json(system=system_retry, user=user)
        print("OLLAMA_JSON_RETRY:", answer_obj)

        if not json_cites_valid(answer_obj, max_k=5):
            return QueryResponse(
                summary="Insufficient evidence found in the current database.",
                recommendation="Model returned invalid JSON or invalid citations. Rejected for safety.",
                evidence_level="N/A",
                confidence_score=float(hits[0]["score"]),
                citations=citations,
            )

    # 7) Return structured response
    return QueryResponse(
        summary=str(answer_obj.get("summary", "")),
        recommendation=str(answer_obj.get("recommendation", "")),
        evidence_level=str(answer_obj.get("evidence_level", "Moderate")),
        confidence_score=float(hits[0]["score"]),
        citations=citations,
    )
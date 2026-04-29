from __future__ import annotations

import json
import re
import urllib.request
from typing import List, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.deps import (
    get_embedder,
    get_eu_collection,
    get_eu_obligations,
    get_settings,
)


router = APIRouter()

RiskCategory = Literal["UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL", "UNKNOWN"]


class ClassifyRequest(BaseModel):
    system_description: str = Field(..., min_length=1)
    ai_use_cases: List[str] = Field(default_factory=list)
    data_subjects: List[str] = Field(default_factory=list)
    industry: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    risk_category: RiskCategory
    annex_iii_match: bool
    matched_annex_iii_items: List[str]
    confidence_score: float
    reasoning: str
    applicable_obligations_count: int


def _build_query_text(payload: ClassifyRequest) -> str:
    parts = [
        f"System: {payload.system_description}",
        f"Industry: {payload.industry}",
        f"Use cases: {', '.join(payload.ai_use_cases) if payload.ai_use_cases else 'N/A'}",
        f"Data subjects: {', '.join(payload.data_subjects) if payload.data_subjects else 'N/A'}",
    ]
    return "\n".join(parts)


def _query_annex_chunks(query_text: str, top_k: int = 8) -> list[dict]:
    embedder = get_embedder()
    collection = get_eu_collection()
    query_embedding = embedder.encode([query_text], show_progress_bar=False)
    try:
        payload = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    ids = payload.get("ids", [[]])[0]
    documents = payload.get("documents", [[]])[0]
    metadatas = payload.get("metadatas", [[]])[0]
    distances = payload.get("distances", [[]])[0]

    results: list[dict] = []
    for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        results.append(
            {
                "id": item_id,
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return results


def _extract_annex_matches(results: list[dict]) -> list[str]:
    matches: list[str] = []
    for item in results:
        text = (item.get("document") or "").lower()
        if "annex iii" in text:
            chunk_id = item.get("id")
            article = (item.get("metadata") or {}).get("article")
            if article:
                matches.append(f"Article {article} - {chunk_id}")
            else:
                matches.append(str(chunk_id))
    return matches


def _call_gemini(model: str, api_key: str, prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))

    candidates = body.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        raise ValueError("Gemini returned empty content")
    return str(parts[0].get("text") or "")


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    return {}


def _normalize_risk_category(value: str) -> RiskCategory:
    normalized = value.strip().upper()
    if normalized in {"UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"}:
        return normalized  # type: ignore[return-value]
    return "UNKNOWN"


def _count_applicable_obligations(risk_category: RiskCategory) -> int:
    if risk_category == "UNKNOWN":
        return 0
    target = risk_category.lower()
    obligations = get_eu_obligations()
    return sum(
        1
        for item in obligations
        if (item.get("applies_to_risk_tier") or "").lower() in {target, "all"}
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_system(payload: ClassifyRequest) -> ClassifyResponse:
    settings = get_settings()
    query_text = _build_query_text(payload)
    annex_results = _query_annex_chunks(query_text)
    matched_annex_items = _extract_annex_matches(annex_results)
    annex_match = bool(matched_annex_items)

    if not settings.GEMINI_API_KEY:
        return ClassifyResponse(
            risk_category="UNKNOWN",
            annex_iii_match=annex_match,
            matched_annex_iii_items=matched_annex_items,
            confidence_score=0.0,
            reasoning="LLM not configured",
            applicable_obligations_count=0,
        )

    prompt = "\n".join(
        [
            "You are an EU AI Act compliance analyst.",
            "Classify the system into one of these risk categories: UNACCEPTABLE, HIGH, LIMITED, MINIMAL.",
            "Return JSON only with keys: risk_category, confidence_score, reasoning.",
            "confidence_score must be between 0 and 1.",
            "System details:",
            query_text,
            f"Annex III matches: {matched_annex_items or 'None'}",
        ]
    )

    try:
        response_text = _call_gemini(settings.GEMINI_MODEL, settings.GEMINI_API_KEY, prompt)
        payload_json = _extract_json(response_text)
        risk_category = _normalize_risk_category(str(payload_json.get("risk_category", "")))
        confidence_score = float(payload_json.get("confidence_score", 0.5))
        confidence_score = max(0.0, min(confidence_score, 1.0))
        reasoning = str(payload_json.get("reasoning", "")) or "Classification completed."
    except Exception as exc:
        return ClassifyResponse(
            risk_category="UNKNOWN",
            annex_iii_match=annex_match,
            matched_annex_iii_items=matched_annex_items,
            confidence_score=0.0,
            reasoning=f"Classification failed: {exc}",
            applicable_obligations_count=0,
        )

    return ClassifyResponse(
        risk_category=risk_category,
        annex_iii_match=annex_match,
        matched_annex_iii_items=matched_annex_items,
        confidence_score=confidence_score,
        reasoning=reasoning,
        applicable_obligations_count=_count_applicable_obligations(risk_category),
    )

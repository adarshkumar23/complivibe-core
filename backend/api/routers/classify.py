from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.deps import (
    get_embedder,
    get_eu_collection,
    get_eu_obligations,
    get_settings,
)


router = APIRouter()

ANNEX_III_ITEMS = [
    "Biometric identification and categorisation of natural persons",
    "Management and operation of critical infrastructure",
    "Education and vocational training",
    "Employment, worker management and access to self-employment",
    "Access to essential private and public services and benefits",
    "Law enforcement",
    "Migration, asylum and border control management",
    "Administration of justice and democratic processes",
]
ANNEX_III_LOOKUP = {item.lower(): item for item in ANNEX_III_ITEMS}
HIGH_RISK_ARTICLES = {"5", "6", "7", "8", "9", "10"}
RISK_CATEGORIES = {"UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL", "UNKNOWN"}
PROHIBITED_RISK_ITEMS = [
    "Social scoring by public authorities",
    "Real-time remote biometric identification in public spaces for law enforcement",
    "Subliminal manipulation causing harm",
    "Exploitation of vulnerabilities of specific groups",
]


class ClassifyRequest(BaseModel):
    system_description: str = Field(..., min_length=1)
    ai_use_cases: list[str] = Field(default_factory=list)
    data_subjects: list[str] = Field(default_factory=list)
    industry: str = ""


class ClassifyResponse(BaseModel):
    risk_category: str
    annex_iii_match: bool
    matched_annex_iii_items: list[str]
    confidence_score: float
    reasoning: str
    applicable_obligations_count: int
    retrieved_chunks_count: int


def _build_query_text(payload: ClassifyRequest) -> str:
    parts = [
        f"System description: {payload.system_description}",
        f"Industry: {payload.industry or 'N/A'}",
        f"AI use cases: {', '.join(payload.ai_use_cases) if payload.ai_use_cases else 'N/A'}",
        f"Data subjects: {', '.join(payload.data_subjects) if payload.data_subjects else 'N/A'}",
    ]
    return "\n".join(parts)


def _safe_strip(value: Any) -> str:
    return str(value or "").strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    # Strip markdown code fences if present
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            loaded = json.loads(match.group(0))
            return loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_risk_category(value: Any) -> str:
    normalized = _safe_strip(value).upper()
    if normalized in RISK_CATEGORIES:
        return normalized
    aliases = {
        "UNACCEPTABLE RISK": "UNACCEPTABLE",
        "HIGH RISK": "HIGH",
        "LIMITED RISK": "LIMITED",
        "MINIMAL RISK": "MINIMAL",
        "LOW RISK": "MINIMAL",
        "HIGH-RISK": "HIGH",
        "UNACCEPTABLE-RISK": "UNACCEPTABLE",
    }
    return aliases.get(normalized, "UNKNOWN")


def _normalize_annex_items(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []

    normalized_items: list[str] = []
    seen: set[str] = set()
    for item in items:
        candidate = _safe_strip(item)
        if not candidate:
            continue
        resolved = ANNEX_III_LOOKUP.get(candidate.lower())
        if resolved and resolved not in seen:
            seen.add(resolved)
            normalized_items.append(resolved)
    return normalized_items


def _count_applicable_obligations(risk_category: str) -> int:
    if risk_category == "UNKNOWN":
        return 0

    target = risk_category.strip().lower()
    obligations = get_eu_obligations()
    count = 0
    for item in obligations:
        applies_to = _safe_strip(item.get("applies_to_risk_tier")).lower()
        if applies_to in {target, "all", "any"}:
            count += 1
            continue
        if target == "unacceptable" and applies_to in {"prohibited", "prohibition", "prohibited risk"}:
            count += 1
            continue
        if target == "high" and applies_to in {"high risk", "high-risk"}:
            count += 1
    return count


def _retrieve_relevant_chunks(query_text: str) -> list[dict[str, Any]]:
    try:
        embedder = get_embedder()
        collection = get_eu_collection()
        query_embedding = embedder.encode([query_text], show_progress_bar=False).tolist()
        payload = collection.query(
            query_embeddings=query_embedding,
            n_results=15,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    ids = payload.get("ids", [[]])
    documents = payload.get("documents", [[]])
    metadatas = payload.get("metadatas", [[]])
    distances = payload.get("distances", [[]])

    results: list[dict[str, Any]] = []
    for item_id, document, metadata, distance in zip(ids[0], documents[0], metadatas[0], distances[0]):
        article = _safe_strip((metadata or {}).get("article"))
        if article not in HIGH_RISK_ARTICLES:
            continue
        results.append(
            {
                "id": item_id,
                "document": _safe_strip(document),
                "metadata": metadata or {},
                "distance": distance,
                "article": article,
            }
        )
    return results


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "[]"

    lines: list[str] = []
    for chunk in chunks:
        lines.append(
            json.dumps(
                {
                    "id": chunk.get("id"),
                    "article": chunk.get("article"),
                    "distance": chunk.get("distance"),
                    "document": chunk.get("document"),
                    "metadata": chunk.get("metadata"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def _build_prompt(payload: ClassifyRequest, chunks: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "You are an EU AI Act compliance classifier.",
            "Analyze the system description and the retrieved EU AI Act context.",
            "Use the Annex III items and prohibited-risk indicators below to decide the classification.",
            "If the system clearly fits one of the prohibited indicators, return risk_category as UNACCEPTABLE.",
            "Return ONLY valid JSON with exactly these keys: risk_category, matched_annex_iii_items, confidence_score, reasoning.",
            "risk_category must be one of: UNACCEPTABLE, HIGH, LIMITED, MINIMAL, UNKNOWN.",
            "matched_annex_iii_items must only contain exact strings from the provided Annex III list.",
            "confidence_score must be a number between 0 and 1.",
            "reasoning must be a concise explanation.",
            "",
            "System description:",
            payload.system_description,
            "",
            f"Industry: {payload.industry or 'N/A'}",
            f"AI use cases: {', '.join(payload.ai_use_cases) if payload.ai_use_cases else 'N/A'}",
            f"Data subjects: {', '.join(payload.data_subjects) if payload.data_subjects else 'N/A'}",
            "",
            "Retrieved EU AI Act chunks:",
            _format_chunks_for_prompt(chunks),
            "",
            "Annex III categories:",
            *[f"- {item}" for item in ANNEX_III_ITEMS],
            "",
            "Prohibited indicators:",
            *[f"- {item}" for item in PROHIBITED_RISK_ITEMS],
        ]
    )


def _call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    try:
        from google import generativeai as genai
    except Exception as exc:
        raise RuntimeError(f"Gemini SDK unavailable: {exc}") from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 8192,
        },
    )

    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        if parts:
            first_part = parts[0]
            return str(getattr(first_part, "text", "") or "")

    raise RuntimeError("Gemini returned an empty response")


def _build_unknown_response(
    *,
    reasoning: str,
    retrieved_chunks_count: int,
) -> ClassifyResponse:
    return ClassifyResponse(
        risk_category="UNKNOWN",
        annex_iii_match=False,
        matched_annex_iii_items=[],
        confidence_score=0.0,
        reasoning=reasoning,
        applicable_obligations_count=0,
        retrieved_chunks_count=retrieved_chunks_count,
    )


@router.post("/classify", response_model=ClassifyResponse)
def classify_system(payload: ClassifyRequest) -> ClassifyResponse:
    settings = get_settings()
    query_text = _build_query_text(payload)
    retrieved_chunks = _retrieve_relevant_chunks(query_text)

    if not settings.GEMINI_API_KEY:
        return _build_unknown_response(
            reasoning="LLM not configured",
            retrieved_chunks_count=len(retrieved_chunks),
        )

    prompt = _build_prompt(payload, retrieved_chunks)

    try:
        response_text = _call_gemini(settings.GEMINI_API_KEY, settings.GEMINI_MODEL, prompt)
        payload_json = _extract_json_object(response_text)
        risk_category = _normalize_risk_category(payload_json.get("risk_category"))
        matched_annex_iii_items = _normalize_annex_items(payload_json.get("matched_annex_iii_items"))

        confidence_score_raw = payload_json.get("confidence_score", 0.0)
        try:
            confidence_score = float(confidence_score_raw)
        except (TypeError, ValueError):
            confidence_score = 0.0
        confidence_score = max(0.0, min(confidence_score, 1.0))

        reasoning = _safe_strip(payload_json.get("reasoning"))
        if not reasoning:
            reasoning = "Classification completed."

        return ClassifyResponse(
            risk_category=risk_category,
            annex_iii_match=bool(matched_annex_iii_items),
            matched_annex_iii_items=matched_annex_iii_items,
            confidence_score=confidence_score,
            reasoning=reasoning,
            applicable_obligations_count=_count_applicable_obligations(risk_category),
            retrieved_chunks_count=len(retrieved_chunks),
        )
    except Exception as exc:
        return _build_unknown_response(
            reasoning=f"Classification failed: {exc}",
            retrieved_chunks_count=len(retrieved_chunks),
        )

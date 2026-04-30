from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import (
    get_embedder,
    get_eu_collection,
    get_eu_obligations,
    get_settings,
)

logger = logging.getLogger("uvicorn")

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory document store (session-scoped)
# ---------------------------------------------------------------------------
_document_store: dict[str, "AnnexIVResponse"] = {}

# ---------------------------------------------------------------------------
# Annex IV section definitions
# ---------------------------------------------------------------------------
ANNEX_IV_SECTIONS: list[tuple[int, str]] = [
    (1, "General description of the AI system"),
    (2, "Description of the development process"),
    (3, "Monitoring, functioning and control of the AI system"),
    (4, "Robustness, accuracy and cybersecurity measures"),
    (5, "Data governance and data management"),
    (6, "Technical capabilities and limitations"),
    (7, "Risk management system"),
    (8, "Post-market monitoring plan"),
]

SECTION_PLACEHOLDER = (
    "This section could not be generated automatically. "
    "Please complete it manually in accordance with Annex IV of the EU AI Act."
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AnnexIVRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    system_name: str = Field(..., min_length=1)
    system_description: str = Field(..., min_length=1)
    risk_category: str = Field(..., pattern=r"^(HIGH|UNACCEPTABLE)$")
    industry: str
    ai_use_cases: list[str] = Field(default_factory=list)
    data_subjects: list[str] = Field(default_factory=list)
    intended_purpose: str
    development_methodology: str = ""
    training_data_description: str = ""


class AnnexIVSection(BaseModel):
    section_number: int
    section_title: str
    content: str
    obligations_referenced: list[str]
    word_count: int


class AnnexIVResponse(BaseModel):
    document_id: str
    company_name: str
    system_name: str
    risk_category: str
    generated_at: str
    sections: list[AnnexIVSection]
    total_sections: int
    completion_percentage: float


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------


def _retrieve_section_chunks(
    section_title: str,
    system_description: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Embed section title + system description and query ChromaDB top-n chunks."""
    query_text = f"{section_title}\n{system_description}"
    try:
        embedder = get_embedder()
        collection = get_eu_collection()
        embedding = embedder.encode([query_text], show_progress_bar=False).tolist()
        payload = collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.warning("ChromaDB query failed for section '%s': %s", section_title, exc)
        return []

    ids: list[str] = payload.get("ids", [[]])[0]
    documents: list[str] = payload.get("documents", [[]])[0]
    metadatas: list[dict] = payload.get("metadatas", [[]])[0]
    distances: list[float] = payload.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        chunks.append(
            {
                "id": str(chunk_id),
                "document": str(document or "").strip(),
                "metadata": metadata or {},
                "distance": distance,
            }
        )
    return chunks


def _extract_obligation_ids(chunks: list[dict[str, Any]]) -> list[str]:
    """Pull obligation_id (or article) values from chunk metadata."""
    seen: set[str] = set()
    ids: list[str] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        obligation_id = str(meta.get("obligation_id") or meta.get("article") or "").strip()
        if obligation_id and obligation_id not in seen:
            seen.add(obligation_id)
            ids.append(obligation_id)
    return ids


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant regulatory chunks retrieved."
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        article = str(meta.get("article") or "N/A")
        lines.append(f"[Chunk {i} | Article {article}]\n{chunk.get('document', '')}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def _call_gemini(api_key: str, model_name: str, prompt: str) -> str:
    """Call Gemini and return the text response. Raises RuntimeError on failure."""
    try:
        from google import generativeai as genai
    except Exception as exc:
        raise RuntimeError(f"Gemini SDK unavailable: {exc}") from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 8192,
        },
    )

    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        if parts:
            return str(getattr(parts[0], "text", "") or "")

    raise RuntimeError("Gemini returned an empty response")


def _strip_markdown_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from Gemini output."""
    stripped = re.sub(r"^```(?:\w+)?\s*", "", text.strip(), flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _build_section_prompt(
    request: AnnexIVRequest,
    section_number: int,
    section_title: str,
    chunks: list[dict[str, Any]],
) -> str:
    use_cases = ", ".join(request.ai_use_cases) if request.ai_use_cases else "N/A"
    data_subjects = ", ".join(request.data_subjects) if request.data_subjects else "N/A"
    chunks_text = _format_chunks_for_prompt(chunks)

    return "\n".join(
        [
            "You are a technical writer specializing in EU AI Act compliance documentation.",
            f"Generate the content for Section {section_number} of an Annex IV Technical Documentation.",
            "Write 200–300 words of professional, precise technical content for this section.",
            "Ground your response in the retrieved EU AI Act regulatory chunks provided below.",
            "Do NOT use markdown headings, bullet points, or code fences — write plain paragraphs only.",
            "Do NOT include the section title or section number in your output.",
            "",
            "=== SECTION REQUIREMENT ===",
            f"Section {section_number}: {section_title}",
            "",
            "=== COMPANY & SYSTEM CONTEXT ===",
            f"Company: {request.company_name}",
            f"AI System: {request.system_name}",
            f"Risk Category: {request.risk_category}",
            f"Industry: {request.industry}",
            f"System Description: {request.system_description}",
            f"Intended Purpose: {request.intended_purpose}",
            f"AI Use Cases: {use_cases}",
            f"Data Subjects: {data_subjects}",
            f"Development Methodology: {request.development_methodology or 'N/A'}",
            f"Training Data Description: {request.training_data_description or 'N/A'}",
            "",
            "=== RETRIEVED EU AI ACT REGULATORY CONTEXT ===",
            chunks_text,
            "",
            "Now write the section content (200–300 words, plain paragraphs only):",
        ]
    )


# ---------------------------------------------------------------------------
# Section generation
# ---------------------------------------------------------------------------


def _generate_section(
    request: AnnexIVRequest,
    section_number: int,
    section_title: str,
    api_key: str,
    model_name: str,
) -> AnnexIVSection:
    logger.info(
        "Generating Annex IV section %d/%d: '%s'",
        section_number,
        len(ANNEX_IV_SECTIONS),
        section_title,
    )

    chunks = _retrieve_section_chunks(section_title, request.system_description, n_results=5)
    obligation_ids = _extract_obligation_ids(chunks)

    try:
        prompt = _build_section_prompt(request, section_number, section_title, chunks)
        raw_content = _call_gemini(api_key, model_name, prompt)
        content = _strip_markdown_fences(raw_content)
        if not content:
            content = SECTION_PLACEHOLDER
    except Exception as exc:
        logger.warning(
            "Gemini failed for section %d ('%s'): %s — using placeholder.",
            section_number,
            section_title,
            exc,
        )
        content = SECTION_PLACEHOLDER

    word_count = len(content.split())

    logger.info(
        "Section %d complete — %d words, %d obligations referenced.",
        section_number,
        word_count,
        len(obligation_ids),
    )

    return AnnexIVSection(
        section_number=section_number,
        section_title=section_title,
        content=content,
        obligations_referenced=obligation_ids,
        word_count=word_count,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=AnnexIVResponse)
def generate_document(request: AnnexIVRequest) -> AnnexIVResponse:
    """Generate an Annex IV Technical Documentation for a high-risk AI system."""
    settings = get_settings()

    # Ensure obligations are loaded (validates data path on startup)
    get_eu_obligations()

    document_id = str(uuid.uuid4())
    generated_at = datetime.now(tz=timezone.utc).isoformat()

    logger.info(
        "Starting Annex IV generation — document_id=%s company='%s' system='%s'",
        document_id,
        request.company_name,
        request.system_name,
    )

    sections: list[AnnexIVSection] = []

    if not settings.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not configured — all sections will use placeholder text.")

    for section_number, section_title in ANNEX_IV_SECTIONS:
        if settings.GEMINI_API_KEY:
            section = _generate_section(
                request,
                section_number,
                section_title,
                settings.GEMINI_API_KEY,
                settings.GEMINI_MODEL,
            )
        else:
            # No API key configured — emit placeholders without calling Gemini
            chunks = _retrieve_section_chunks(section_title, request.system_description)
            obligation_ids = _extract_obligation_ids(chunks)
            section = AnnexIVSection(
                section_number=section_number,
                section_title=section_title,
                content=SECTION_PLACEHOLDER,
                obligations_referenced=obligation_ids,
                word_count=len(SECTION_PLACEHOLDER.split()),
            )
        sections.append(section)

    successful = sum(1 for s in sections if s.content != SECTION_PLACEHOLDER)
    completion_percentage = round((successful / len(sections)) * 100.0, 2) if sections else 0.0

    response = AnnexIVResponse(
        document_id=document_id,
        company_name=request.company_name,
        system_name=request.system_name,
        risk_category=request.risk_category,
        generated_at=generated_at,
        sections=sections,
        total_sections=len(sections),
        completion_percentage=completion_percentage,
    )

    _document_store[document_id] = response

    logger.info(
        "Annex IV generation complete — document_id=%s completion=%.1f%%",
        document_id,
        completion_percentage,
    )

    return response


@router.get("/{document_id}", response_model=AnnexIVResponse)
def get_document(document_id: str) -> AnnexIVResponse:
    """Retrieve a previously generated Annex IV document by its ID."""
    doc = _document_store.get(document_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found. Documents are stored for the current session only.",
        )
    return doc

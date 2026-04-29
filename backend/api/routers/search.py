from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.api.deps import get_dpdp_collection, get_embedder, get_eu_collection


router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    source: Literal["eu_ai_act", "dpdp", "all"] = "all"
    top_k: int = 10


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    article: str | None
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


def _query_collection(collection, embedding, top_k: int, source: str) -> list[SearchResult]:
    try:
        payload = collection.query(
            query_embeddings=embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    ids = payload.get("ids", [[]])[0]
    documents = payload.get("documents", [[]])[0]
    metadatas = payload.get("metadatas", [[]])[0]
    distances = payload.get("distances", [[]])[0]

    results: list[SearchResult] = []
    for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        score = 1.0 - float(distance or 0.0)
        results.append(
            SearchResult(
                chunk_id=str(item_id),
                text=str(document or ""),
                source=source,
                article=(metadata or {}).get("article"),
                score=score,
            )
        )
    return results


@router.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest) -> SearchResponse:
    embedder = get_embedder()
    embedding = embedder.encode([payload.query], show_progress_bar=False)
    results: list[SearchResult] = []

    if payload.source in {"eu_ai_act", "all"}:
        results.extend(_query_collection(get_eu_collection(), embedding, payload.top_k, "eu_ai_act"))
    if payload.source in {"dpdp", "all"}:
        results.extend(_query_collection(get_dpdp_collection(), embedding, payload.top_k, "dpdp"))

    results.sort(key=lambda item: item.score, reverse=True)
    return SearchResponse(results=results[: payload.top_k])

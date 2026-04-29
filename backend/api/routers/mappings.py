from __future__ import annotations

import re
from typing import List, Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from backend.api.deps import get_cross_mappings


router = APIRouter()


class MappingItem(BaseModel):
    map_id: str
    eu_obligation_id: str
    dpdp_obligation_id: str
    eu_obligation_type: Optional[str] = None
    dpdp_obligation_type: Optional[str] = None
    relationship: Optional[str] = None
    overlap_score: Optional[float] = None
    eu_satisfied_by_dpdp: Optional[bool] = None
    dpdp_satisfied_by_eu: Optional[bool] = None
    unified_action: Optional[str] = None
    eu_additional_requirements: Optional[str] = None
    dpdp_additional_requirements: Optional[str] = None
    confidence: Optional[float] = None
    mapping_method: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class MappingsResponse(BaseModel):
    total: int
    items: List[MappingItem]


def _extract_id_part(value: str, token: str) -> Optional[str]:
    match = re.search(rf"{token}(\d+[a-zA-Z]?)", value)
    if not match:
        return None
    return match.group(1)


@router.get("/mappings", response_model=MappingsResponse)
def list_mappings(
    eu_article: Optional[str] = None,
    dpdp_section: Optional[str] = None,
    min_similarity: float = Query(0.3, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> MappingsResponse:
    mappings = get_cross_mappings()
    filtered: list[dict] = []

    for item in mappings:
        overlap_score = float(item.get("overlap_score") or 0.0)
        if overlap_score < min_similarity:
            continue
        if eu_article:
            article_value = _extract_id_part(str(item.get("eu_obligation_id") or ""), "art")
            if not article_value or article_value.lower() != eu_article.lower():
                continue
        if dpdp_section:
            section_value = _extract_id_part(str(item.get("dpdp_obligation_id") or ""), "sec")
            if not section_value or section_value.lower() != dpdp_section.lower():
                continue
        filtered.append(item)

    total = len(filtered)
    items = filtered[offset : offset + limit]
    return MappingsResponse(total=total, items=items)

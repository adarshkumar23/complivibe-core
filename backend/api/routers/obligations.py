from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict

from backend.api.deps import get_dpdp_obligations, get_eu_obligations


router = APIRouter()


class ObligationItem(BaseModel):
    obligation_id: str
    regulation: Optional[str] = None
    regulation_display: Optional[str] = None
    article: Optional[str] = None
    chapter: Optional[str] = None
    obligation_type: Optional[str] = None
    who_must_comply: Optional[str] = None
    who_must_comply_original: Optional[str] = None
    what_must_be_done: Optional[str] = None
    legal_basis: Optional[str] = None
    applies_to_risk_tier: Optional[str] = None
    deadline: Optional[str] = None
    penalty_max_eur: Optional[int] = None
    penalty_max_inr: Optional[int] = None
    source_chunk_id: Optional[str] = None
    confidence: Optional[float] = None
    raw_text: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class ObligationsResponse(BaseModel):
    total: int
    items: List[ObligationItem]
    source: str


class ChecklistItem(BaseModel):
    id: str
    article: Optional[str]
    obligation_text: Optional[str]
    category: Optional[str]
    completed: bool = False


class ChecklistResponse(BaseModel):
    checklist: List[ChecklistItem]


def _filter_obligations(
    obligations: list[dict],
    risk_category: Optional[str],
    article: Optional[str],
) -> list[dict]:
    filtered = obligations
    if risk_category:
        target = risk_category.lower()
        filtered = [
            item
            for item in filtered
            if (item.get("applies_to_risk_tier") or "").lower() in {target, "all"}
        ]
    if article:
        filtered = [
            item for item in filtered if str(item.get("article") or "").lower() == article.lower()
        ]
    return filtered


@router.get("/obligations", response_model=ObligationsResponse)
def list_obligations(
    source: Literal["eu_ai_act", "dpdp", "all"] = "all",
    risk_category: Optional[str] = None,
    article: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> ObligationsResponse:
    obligations: list[dict] = []
    if source in {"eu_ai_act", "all"}:
        obligations.extend(get_eu_obligations())
    if source in {"dpdp", "all"}:
        obligations.extend(get_dpdp_obligations())

    filtered = _filter_obligations(obligations, risk_category, article)
    total = len(filtered)
    items = filtered[offset : offset + limit]
    return ObligationsResponse(total=total, items=items, source=source)


@router.get("/obligations/checklist", response_model=ChecklistResponse)
def obligations_checklist(
    risk_category: str = Query(..., min_length=1),
    source: Literal["eu_ai_act", "dpdp", "all"] = "all",
) -> ChecklistResponse:
    obligations: list[dict] = []
    if source in {"eu_ai_act", "all"}:
        obligations.extend(get_eu_obligations())
    if source in {"dpdp", "all"}:
        obligations.extend(get_dpdp_obligations())

    filtered = _filter_obligations(obligations, risk_category, None)
    checklist = [
        ChecklistItem(
            id=str(item.get("obligation_id")),
            article=item.get("article"),
            obligation_text=item.get("what_must_be_done"),
            category=item.get("obligation_type"),
            completed=False,
        )
        for item in filtered
    ]
    return ChecklistResponse(checklist=checklist)

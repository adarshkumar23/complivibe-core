from __future__ import annotations

from typing import Dict, List, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.api.deps import get_cross_mappings, get_dpdp_obligations, get_eu_obligations


router = APIRouter()


class ScoreResponse(BaseModel):
    overall_score: float
    breakdown: Dict[str, float]
    risk_level: str
    total_obligations: int
    mapped_obligations: int
    recommendations: List[str]


def _risk_level(score: float) -> str:
    if score >= 80:
        return "LOW"
    if score >= 60:
        return "MEDIUM"
    return "HIGH"


@router.get("/score", response_model=ScoreResponse)
def get_score(
    company_profile_id: str = Query(..., min_length=1),
    source: Literal["eu_ai_act", "dpdp", "all"] = "all",
) -> ScoreResponse:
    eu_obligations = get_eu_obligations()
    dpdp_obligations = get_dpdp_obligations()
    mappings = get_cross_mappings()

    if source == "eu_ai_act":
        total_obligations = len(eu_obligations)
        mapped_obligations = len({item.get("eu_obligation_id") for item in mappings})
    elif source == "dpdp":
        total_obligations = len(dpdp_obligations)
        mapped_obligations = len({item.get("dpdp_obligation_id") for item in mappings})
    else:
        total_obligations = len(eu_obligations) + len(dpdp_obligations)
        mapped_obligations = len(mappings)

    if total_obligations <= 0:
        overall_score = 0.0
    else:
        coverage = mapped_obligations / total_obligations
        overall_score = round(60.0 + min(coverage, 1.0) * 40.0, 2)

    breakdown = {
        "governance": 72.0,
        "data_protection": 68.0,
        "documentation": 65.0,
    }

    recommendations = [
        "Prioritize high-risk obligations first.",
        "Establish evidence collection for mapped obligations.",
        "Review data protection controls quarterly.",
    ]

    return ScoreResponse(
        overall_score=overall_score,
        breakdown=breakdown,
        risk_level=_risk_level(overall_score),
        total_obligations=total_obligations,
        mapped_obligations=mapped_obligations,
        recommendations=recommendations,
    )

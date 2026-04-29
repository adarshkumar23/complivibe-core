from __future__ import annotations

from typing import List
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter()


class CompanyProfileRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    industry: str
    product_description: str
    uses_ai: bool
    processes_eu_personal_data: bool
    processes_india_personal_data: bool
    employee_count: int
    annual_revenue_usd: float
    ai_use_cases: List[str] = Field(default_factory=list)
    data_subjects: List[str] = Field(default_factory=list)


class CompanyProfileResponse(BaseModel):
    profile_id: str
    company_name: str
    applicable_regulations: List[str]
    next_step: str


@router.post("/company/profile", response_model=CompanyProfileResponse)
def create_company_profile(payload: CompanyProfileRequest) -> CompanyProfileResponse:
    applicable: List[str] = []
    if payload.uses_ai or payload.processes_eu_personal_data:
        applicable.append("eu_ai_act")
    if payload.processes_india_personal_data:
        applicable.append("dpdp")

    next_step = "Run AI system classification to determine risk tier."
    return CompanyProfileResponse(
        profile_id=str(uuid4()),
        company_name=payload.company_name,
        applicable_regulations=applicable,
        next_step=next_step,
    )

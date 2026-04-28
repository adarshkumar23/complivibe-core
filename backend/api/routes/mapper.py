"""
Cross-mapper API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.mapper.cross_mapper import CrossMapper, MappingEntry

router = APIRouter()
_mapper = CrossMapper()


class MappingEntryOut(BaseModel):
    eu_ai_act_ref: str
    dpdp_ref: str
    theme: str
    notes: str


class MappingResultOut(BaseModel):
    query_ref: str
    source_regulation: str
    matches: list[MappingEntryOut]


def _entry_to_out(entry: MappingEntry) -> MappingEntryOut:
    return MappingEntryOut(
        eu_ai_act_ref=entry.eu_ai_act_ref,
        dpdp_ref=entry.dpdp_ref,
        theme=entry.theme,
        notes=entry.notes,
    )


@router.get("/eu-to-dpdp", response_model=MappingResultOut)
def eu_to_dpdp(ref: str) -> MappingResultOut:
    """Return DPDP provisions that correspond to an EU AI Act reference."""
    result = _mapper.find_dpdp_for_eu(ref)
    return MappingResultOut(
        query_ref=result.query_ref,
        source_regulation=result.source_regulation,
        matches=[_entry_to_out(m) for m in result.matches],
    )


@router.get("/dpdp-to-eu", response_model=MappingResultOut)
def dpdp_to_eu(ref: str) -> MappingResultOut:
    """Return EU AI Act provisions that correspond to a DPDP reference."""
    result = _mapper.find_eu_for_dpdp(ref)
    return MappingResultOut(
        query_ref=result.query_ref,
        source_regulation=result.source_regulation,
        matches=[_entry_to_out(m) for m in result.matches],
    )


@router.get("/all", response_model=list[MappingEntryOut])
def all_mappings() -> list[MappingEntryOut]:
    """Return all available cross-regulation mappings."""
    return [_entry_to_out(m) for m in _mapper.all_mappings()]

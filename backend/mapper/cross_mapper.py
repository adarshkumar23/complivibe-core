"""
EU AI Act ↔ India DPDP cross-mapper (Week 3-4).

Maps obligations and requirements between the EU AI Act and India's
Digital Personal Data Protection (DPDP) Act 2023.

The mapping tables are loaded from ``data/mappings/`` (CSV/JSON).
This module provides a programmatic interface over those tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MappingEntry:
    """A single cross-regulation mapping between two provisions."""

    eu_ai_act_ref: str
    dpdp_ref: str
    theme: str
    notes: str = ""


@dataclass
class MappingResult:
    """The result of a cross-mapping lookup."""

    query_ref: str
    source_regulation: str
    matches: list[MappingEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in seed mappings (subset — full tables live in data/mappings/)
# ---------------------------------------------------------------------------

SEED_MAPPINGS: list[MappingEntry] = [
    MappingEntry(
        eu_ai_act_ref="Art. 10 (Data governance)",
        dpdp_ref="Sec. 4 (Grounds for processing)",
        theme="data_quality_and_governance",
        notes="Both require lawful basis and data quality assurance.",
    ),
    MappingEntry(
        eu_ai_act_ref="Art. 13 (Transparency)",
        dpdp_ref="Sec. 6 (Notice)",
        theme="transparency_and_notice",
        notes="Transparency obligations for automated decision-making.",
    ),
    MappingEntry(
        eu_ai_act_ref="Art. 14 (Human oversight)",
        dpdp_ref="Sec. 12 (Right to grievance redressal)",
        theme="human_oversight",
        notes="Human review mechanisms align with grievance rights.",
    ),
    MappingEntry(
        eu_ai_act_ref="Art. 9 (Risk management)",
        dpdp_ref="Sec. 8 (Obligations of data fiduciary)",
        theme="risk_management",
        notes="Risk management frameworks required by both regulations.",
    ),
    MappingEntry(
        eu_ai_act_ref="Art. 17 (Quality management system)",
        dpdp_ref="Sec. 8(5) (Accuracy and completeness)",
        theme="data_quality",
        notes="Data quality obligations in AI systems and DPDP fiduciary duties.",
    ),
]


class CrossMapper:
    """
    Cross-maps EU AI Act provisions to DPDP Act provisions and vice-versa.

    Mappings are sourced from the built-in seed table and optionally
    augmented by CSV/JSON files placed in ``data/mappings/``.
    """

    def __init__(self, mappings_dir: Path | str | None = None) -> None:
        self._mappings: list[MappingEntry] = list(SEED_MAPPINGS)
        if mappings_dir is not None:
            self._load_from_dir(Path(mappings_dir))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_dpdp_for_eu(self, eu_ref: str) -> MappingResult:
        """Return DPDP provisions that correspond to an EU AI Act reference."""
        matches = [
            m for m in self._mappings if eu_ref.lower() in m.eu_ai_act_ref.lower()
        ]
        return MappingResult(
            query_ref=eu_ref, source_regulation="EU AI Act", matches=matches
        )

    def find_eu_for_dpdp(self, dpdp_ref: str) -> MappingResult:
        """Return EU AI Act provisions that correspond to a DPDP reference."""
        matches = [
            m for m in self._mappings if dpdp_ref.lower() in m.dpdp_ref.lower()
        ]
        return MappingResult(
            query_ref=dpdp_ref, source_regulation="DPDP", matches=matches
        )

    def all_mappings(self) -> list[MappingEntry]:
        """Return all loaded mappings."""
        return list(self._mappings)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_dir(self, directory: Path) -> None:
        """Load additional mappings from JSON files in *directory*."""
        import json

        if not directory.is_dir():
            return

        for json_file in directory.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                for entry in data:
                    self._mappings.append(
                        MappingEntry(
                            eu_ai_act_ref=entry.get("eu_ai_act_ref", ""),
                            dpdp_ref=entry.get("dpdp_ref", ""),
                            theme=entry.get("theme", ""),
                            notes=entry.get("notes", ""),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                pass

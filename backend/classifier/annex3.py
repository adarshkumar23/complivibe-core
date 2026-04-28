"""
Annex III high-risk AI system classifier (Week 2-3).

Determines whether a described AI system falls into one of the
high-risk categories defined in Annex III of the EU AI Act.

Reference categories (Article 6 / Annex III):
  1. Biometric identification and categorisation
  2. Critical infrastructure management
  3. Education and vocational training
  4. Employment, workers management, access to self-employment
  5. Essential private/public services
  6. Law enforcement
  7. Migration, asylum and border control
  8. Administration of justice and democratic processes
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AnnexIIICategory(str, Enum):
    BIOMETRIC = "biometric_identification"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education"
    EMPLOYMENT = "employment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION = "migration"
    JUSTICE = "justice"
    NOT_HIGH_RISK = "not_high_risk"


@dataclass
class ClassificationResult:
    category: AnnexIIICategory
    is_high_risk: bool
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# Keyword-based heuristic classifier (placeholder until LLM integration)
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[AnnexIIICategory, list[str]] = {
    AnnexIIICategory.BIOMETRIC: [
        "face recognition",
        "fingerprint",
        "iris scan",
        "biometric",
        "voice recognition",
    ],
    AnnexIIICategory.CRITICAL_INFRASTRUCTURE: [
        "power grid",
        "water supply",
        "traffic management",
        "critical infrastructure",
        "energy network",
    ],
    AnnexIIICategory.EDUCATION: [
        "student assessment",
        "exam proctoring",
        "learning analytics",
        "educational",
        "vocational training",
    ],
    AnnexIIICategory.EMPLOYMENT: [
        "recruitment",
        "cv screening",
        "job application",
        "performance monitoring",
        "worker surveillance",
    ],
    AnnexIIICategory.ESSENTIAL_SERVICES: [
        "credit scoring",
        "insurance",
        "social benefit",
        "public service",
        "emergency service",
    ],
    AnnexIIICategory.LAW_ENFORCEMENT: [
        "crime prediction",
        "suspect identification",
        "lie detection",
        "forensic",
        "policing",
    ],
    AnnexIIICategory.MIGRATION: [
        "asylum",
        "border control",
        "visa application",
        "migration",
        "refugee",
    ],
    AnnexIIICategory.JUSTICE: [
        "legal decision",
        "court",
        "sentencing",
        "judicial",
        "electoral",
    ],
}


class AnnexIIIClassifier:
    """
    Classifies an AI system description against EU AI Act Annex III.

    This implementation uses keyword heuristics as a baseline.
    It is designed to be replaced or augmented with an LLM-based
    classifier in a later sprint.
    """

    def classify(self, description: str) -> ClassificationResult:
        """
        Classify *description* and return a :class:`ClassificationResult`.

        Args:
            description: Plain-text description of the AI system.

        Returns:
            A classification result with category, risk flag, confidence,
            and a brief rationale.
        """
        description_lower = description.lower()
        scores: dict[AnnexIIICategory, int] = {}

        for category, keywords in _CATEGORY_KEYWORDS.items():
            hit_count = sum(1 for kw in keywords if kw in description_lower)
            if hit_count:
                scores[category] = hit_count

        if not scores:
            return ClassificationResult(
                category=AnnexIIICategory.NOT_HIGH_RISK,
                is_high_risk=False,
                confidence=0.9,
                rationale="No high-risk keywords detected.",
            )

        best_category = max(scores, key=lambda c: scores[c])
        best_score = scores[best_category]
        total_keywords = len(_CATEGORY_KEYWORDS[best_category])
        confidence = min(best_score / total_keywords, 1.0)

        return ClassificationResult(
            category=best_category,
            is_high_risk=True,
            confidence=confidence,
            rationale=(
                f"Matched {best_score} keyword(s) for category "
                f"'{best_category.value}'."
            ),
        )

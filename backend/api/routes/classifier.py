"""
Classifier API routes.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from backend.classifier.annex3 import AnnexIIIClassifier, ClassificationResult

router = APIRouter()
_classifier = AnnexIIIClassifier()


class ClassifyRequest(BaseModel):
    description: str


class ClassifyResponse(BaseModel):
    category: str
    is_high_risk: bool
    confidence: float
    rationale: str


@router.post("/classify", response_model=ClassifyResponse)
def classify_system(request: ClassifyRequest) -> ClassifyResponse:
    """
    Classify an AI system description against EU AI Act Annex III.

    Args:
        request: Request body containing the system description.
    """
    result: ClassificationResult = _classifier.classify(request.description)
    return ClassifyResponse(
        category=result.category.value,
        is_high_risk=result.is_high_risk,
        confidence=result.confidence,
        rationale=result.rationale,
    )

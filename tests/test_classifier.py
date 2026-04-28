"""
Tests for the Annex III classifier.
"""

from backend.classifier.annex3 import AnnexIIICategory, AnnexIIIClassifier


class TestAnnexIIIClassifier:
    def setup_method(self):
        self.classifier = AnnexIIIClassifier()

    def test_biometric_detection(self):
        desc = "A face recognition system for airport security screening."
        result = self.classifier.classify(desc)
        assert result.is_high_risk is True
        assert result.category == AnnexIIICategory.BIOMETRIC

    def test_employment_detection(self):
        desc = "An automated recruitment system for CV screening and job application ranking."
        result = self.classifier.classify(desc)
        assert result.is_high_risk is True
        assert result.category == AnnexIIICategory.EMPLOYMENT

    def test_not_high_risk(self):
        desc = "A simple spell-checker for word processing documents."
        result = self.classifier.classify(desc)
        assert result.is_high_risk is False
        assert result.category == AnnexIIICategory.NOT_HIGH_RISK

    def test_confidence_between_0_and_1(self):
        desc = "Biometric iris scan system for border control."
        result = self.classifier.classify(desc)
        assert 0.0 <= result.confidence <= 1.0

    def test_rationale_non_empty(self):
        desc = "AI-powered crime prediction tool used by law enforcement agencies."
        result = self.classifier.classify(desc)
        assert len(result.rationale) > 0

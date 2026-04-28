"""
Tests for the EU AI Act ↔ DPDP cross-mapper.
"""

from backend.mapper.cross_mapper import CrossMapper


class TestCrossMapper:
    def setup_method(self):
        self.mapper = CrossMapper()

    def test_find_dpdp_for_eu_transparency(self):
        result = self.mapper.find_dpdp_for_eu("Art. 13")
        assert result.source_regulation == "EU AI Act"
        assert len(result.matches) > 0
        assert any("Sec. 6" in m.dpdp_ref for m in result.matches)

    def test_find_eu_for_dpdp_notice(self):
        result = self.mapper.find_eu_for_dpdp("Sec. 6")
        assert result.source_regulation == "DPDP"
        assert len(result.matches) > 0
        assert any("Art. 13" in m.eu_ai_act_ref for m in result.matches)

    def test_no_match_returns_empty(self):
        result = self.mapper.find_dpdp_for_eu("Art. 999")
        assert result.matches == []

    def test_all_mappings_non_empty(self):
        mappings = self.mapper.all_mappings()
        assert len(mappings) > 0

    def test_seed_mappings_have_required_fields(self):
        for m in self.mapper.all_mappings():
            assert m.eu_ai_act_ref
            assert m.dpdp_ref
            assert m.theme

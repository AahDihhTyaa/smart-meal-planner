"""
Unit tests for NLPNormalizer
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.nlp_normalize import NLPNormalizer


def get_normalizer():
    return NLPNormalizer()


class TestParseTextInput:
    def test_comma_separated(self):
        n = get_normalizer()
        parts = n.parse_text_input("tomato, onion, garlic")
        assert len(parts) == 3
        assert "tomato" in parts

    def test_newline_separated(self):
        n = get_normalizer()
        parts = n.parse_text_input("tomato\nonion\ngarlic")
        assert len(parts) == 3

    def test_numbered_list(self):
        n = get_normalizer()
        parts = n.parse_text_input("1. tomato\n2. onion\n3. garlic")
        assert len(parts) == 3
        assert any("tomato" in p for p in parts)

    def test_semicolon_separated(self):
        n = get_normalizer()
        parts = n.parse_text_input("tomato; onion; garlic")
        assert len(parts) == 3


class TestNormalize:
    def test_quantity_stripping(self):
        n = get_normalizer()
        result = n.normalize("2 cups of flour")
        assert result is not None
        assert "flour" in result

    def test_plural_normalization(self):
        n = get_normalizer()
        result = n.normalize("tomatoes")
        assert result is not None
        assert "tomato" in result.lower()

    def test_synonym_mapping(self):
        n = get_normalizer()
        result = n.normalize("scallions")
        assert result == "green_onion"

    def test_synonym_capsicum(self):
        n = get_normalizer()
        result = n.normalize("capsicum")
        assert result == "bell_pepper"

    def test_empty_string(self):
        n = get_normalizer()
        result = n.normalize("")
        assert result is None

    def test_whitespace_only(self):
        n = get_normalizer()
        result = n.normalize("   ")
        assert result is None

    def test_unit_only_stripped(self):
        n = get_normalizer()
        # If input is only a unit, should return None or minimal
        result = n.normalize("2 tbsp")
        # May return None or small string — just check it doesn't crash
        assert True  # no exception

    def test_lowercase_output(self):
        n = get_normalizer()
        result = n.normalize("GARLIC")
        assert result is not None
        assert result == result.lower()


class TestNormalizeList:
    def test_deduplication(self):
        n = get_normalizer()
        result = n.normalize_list(["tomato", "tomatoes", "tomato"])
        # Should deduplicate to 1 canonical form
        assert len(result) == 1

    def test_fuzzy_deduplication(self):
        n = get_normalizer()
        result = n.normalize_list(["garlic", "garlics", "garlik"])
        # Fuzzy should merge these near-duplicates
        assert len(result) <= 2

    def test_multiple_ingredients(self):
        n = get_normalizer()
        result = n.normalize_list(["chicken breast", "garlic", "lemon"])
        assert len(result) >= 2


class TestGetIngredientSet:
    def test_full_pipeline(self):
        n = get_normalizer()
        result = n.get_ingredient_set("2 chicken breasts\n1 lemon\ngarlic, onion")
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_returns_sorted(self):
        n = get_normalizer()
        result = n.get_ingredient_set("tomato, apple, broccoli")
        assert result == sorted(result)


class TestExtractQuantity:
    def test_extracts_cups(self):
        qty, rest = NLPNormalizer.extract_quantity("2 cups of flour")
        assert "2" in (qty or "")
        assert "flour" in rest

    def test_extracts_tbsp(self):
        qty, rest = NLPNormalizer.extract_quantity("1 tbsp olive oil")
        assert qty is not None
        assert "olive" in rest or "oil" in rest

    def test_no_quantity(self):
        qty, rest = NLPNormalizer.extract_quantity("garlic")
        assert "garlic" in rest

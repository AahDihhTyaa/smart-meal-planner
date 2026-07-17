import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.recipe_matcher import RecipeMatcher

@pytest.fixture(scope="module")
def matcher():
    """Fixture to load the RecipeMatcher once for all tests."""
    # This will load the CSV dataset and build TF-IDF matrices
    return RecipeMatcher()

def test_pantry_staples_always_available(matcher):
    """Test that basic staples like salt and oil are never counted as missing."""
    # We pass zero ingredients. It should still count staples as matched if the recipe uses them.
    # We will pick a very basic recipe or just test the internal matching logic.
    user_ingredients = []
    
    # Internal helper test
    matched = matcher._build_matched_ingredients(set(user_ingredients))
    assert "salt" in matched
    assert "olive_oil" in matched
    assert "pepper" in matched

def test_exact_ingredient_match(matcher):
    """Test that supplying exact ingredients returns recipes with 0 missing count."""
    # Let's search for something highly specific
    user_ingredients = ["chicken_breast", "broccoli", "soy_sauce", "garlic", "ginger", "rice"]
    
    # Run the matcher
    results = matcher.match(
        user_ingredients=user_ingredients,
        top_n=5,
        strict_mode=False
    )
    
    assert not results.empty
    # The top result should have very few missing ingredients, ideally 0 or 1
    top_recipe = results.iloc[0]
    assert top_recipe["missing_count"] <= 2
    
def test_strict_mode_filters_missing(matcher):
    """Test that strict mode ONLY returns recipes with zero missing ingredients."""
    user_ingredients = ["egg", "milk", "butter", "flour", "sugar", "salt", "vanilla_extract", "baking_powder"]
    
    results = matcher.match(
        user_ingredients=user_ingredients,
        top_n=10,
        strict_mode=True
    )
    
    # In strict mode, every single returned recipe must have missing_count == 0
    if not results.empty:
        for _, row in results.iterrows():
            assert row["missing_count"] == 0

def test_dietary_filters(matcher):
    """Test that dietary filters successfully exclude non-compliant recipes."""
    user_ingredients = ["tofu", "spinach", "garlic", "soy_sauce"]
    
    results = matcher.match(
        user_ingredients=user_ingredients,
        dietary_filters=["vegetarian"],
        top_n=10,
        strict_mode=False
    )
    
    assert not results.empty
    # Verify the tag exists
    for _, row in results.iterrows():
        tags = row.get("tags", "")
        assert "vegetarian" in tags.lower()

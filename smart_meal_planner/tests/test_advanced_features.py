import pytest
import pandas as pd
from modules.budget_optimizer import BudgetOptimizer
from modules.recipe_matcher import RecipeMatcher
from modules.vision_input import VisionInputModule

def test_budget_optimizer_costs():
    bo = BudgetOptimizer(daily_budget=10.0)
    
    # Test known ingredient cost
    assert bo.ingredient_cost("chicken breast") > 0
    assert bo.ingredient_cost("salt") < 0.10
    
    # Test recipe cost
    recipe = pd.Series({"ingredients": "chicken breast | salt | olive_oil"})
    cost = bo.recipe_cost(recipe)
    assert cost > 2.0
    
    # Test plan cost
    day_plan = {
        "breakfast": pd.Series({"ingredients": "oats | milk"}),
        "lunch": pd.Series({"ingredients": "rice | beans"}),
        "dinner": pd.Series({"ingredients": "salmon | broccoli"})
    }
    costs = bo.plan_cost(day_plan)
    assert "total" in costs
    assert costs["total"] > 3.0
    
def test_budget_feedback():
    bo = BudgetOptimizer(daily_budget=5.0)
    recipe_costs = {"breakfast": 1.0, "lunch": 2.0, "dinner": 4.5}
    day_plan = {
        "breakfast": pd.Series({"ingredients": "oats"}),
        "lunch": pd.Series({"ingredients": "rice"}),
        "dinner": pd.Series({"ingredients": "salmon"})  # Expensive item
    }
    
    fb = bo.get_budget_feedback(7.5, recipe_costs, day_plan)
    assert fb["over_budget"] is True
    # Should recommend swap for salmon -> tilapia
    assert any("salmon" in s["original"].lower() for s in fb["swaps"])

def test_leftover_matching():
    matcher = RecipeMatcher()
    if matcher._df.empty:
        pytest.skip("recipes.csv not found or empty")
    
    # Let's mock a corpus
    matcher._df = pd.DataFrame([
        {"title": "Chicken Soup", "ingredient_list": ["chicken", "carrot", "onion", "broth"], "tags": ""},
        {"title": "Beef Stew", "ingredient_list": ["beef", "potato", "carrot", "broth"], "tags": ""},
        {"title": "Omelette", "ingredient_list": ["egg", "cheese", "butter"], "tags": ""}
    ])
    
    leftovers = ["chicken", "carrot"]
    df_res = matcher.match_leftovers(leftovers, available_pantry=["onion"], top_n=3)
    
    # Chicken Soup should be top because it uses both leftovers
    assert not df_res.empty
    assert df_res.iloc[0]["title"] == "Chicken Soup"
    assert df_res.iloc[0]["leftover_coverage"] == 1.0

def test_vision_input_fallback_unconfigured():
    vision = VisionInputModule(api_key=None)
    assert not vision.is_available
    data, err = vision.scan_barcode_or_label(b"fake_image_bytes")
    assert data is None
    assert "not configured" in err.lower()

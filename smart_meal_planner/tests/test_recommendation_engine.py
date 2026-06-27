import pytest
import pandas as pd
from modules.recommendation_engine import (
    DifficultyEstimator,
    SeasonalAdvisor,
    ConfidenceAnalyzer,
    KitchenOptimizer,
    RecommendationEngine
)

@pytest.fixture
def sample_recipe_df():
    data = [
        {"title": "Chicken Salad", "ingredient_list": ["chicken_breast", "lettuce", "tomato", "olive_oil"], "cuisine": "american"},
        {"title": "Beef Stir Fry", "ingredient_list": ["beef", "broccoli", "soy_sauce", "garlic", "oil"], "cuisine": "chinese"},
        {"title": "Pasta Pomodoro", "ingredient_list": ["pasta", "tomato", "garlic", "olive_oil", "basil"], "cuisine": "italian"},
        {"title": "Tofu Bowl", "ingredient_list": ["tofu", "rice", "carrot", "broccoli", "soy_sauce"], "cuisine": "asian"},
        {"title": "Omelette", "ingredient_list": ["egg", "butter", "salt", "pepper", "cheese"], "cuisine": "french"},
    ]
    return pd.DataFrame(data)

def test_difficulty_estimator():
    # Salad should be level 1, fast prep
    diff, notes = DifficultyEstimator.estimate("Simple Salad", "Toss everything together", 4)
    assert diff == 1
    assert "No-cook" in notes
    
    # Tempura should be level 5
    diff, notes = DifficultyEstimator.estimate("Tempura Prawns", "Deep fry until golden", 6)
    assert diff == 5
    assert "Professional" in notes

    # High ingredient count bumps difficulty
    diff, notes = DifficultyEstimator.estimate("Simple Soup", "Boil", 15)
    assert diff == 3  # Boil is 2, >12 ingredients = +1

    # Prep time heuristic
    time = DifficultyEstimator.estimate_prep_time(5, 2)  # 10 + (5*3) + (2*8) = 41
    assert time == 41

def test_seasonal_advisor():
    # This might depend on current month, but we can test the lookup logic
    # Assume strawberry is out of season in winter, but let's test a known swap
    swaps = SeasonalAdvisor.get_seasonal_swaps(["tomato"])
    # If it's winter, it suggests canned_tomato. We can just verify it returns a list.
    assert isinstance(swaps, list)
    
    # We can't strictly assert length > 0 because of dynamic season, but we can mock it
    with pytest.MonkeyPatch.context() as m:
        m.setattr(SeasonalAdvisor, "current_season", lambda: "winter")
        swaps = SeasonalAdvisor.get_seasonal_swaps(["tomato", "corn"])
        assert len(swaps) == 2
        assert swaps[0]["swap"] == "canned tomato"
        assert swaps[1]["swap"] == "potato"

def test_confidence_analyzer(sample_recipe_df):
    analyzer = ConfidenceAnalyzer()
    
    # Poor ingredients (F)
    res1 = analyzer.analyze(["salt"], sample_recipe_df)
    assert res1["score"] < 40
    assert res1["grade"] == "F"
    
    # Good ingredients (A)
    res2 = analyzer.analyze(
        ["chicken_breast", "rice", "broccoli", "olive_oil", "garlic", "salt", "pepper", "onion", "apple"], 
        sample_recipe_df
    )
    assert res2["score"] > 80
    assert "protein" in res2["food_groups"] and res2["food_groups"]["protein"]
    assert "carbs" in res2["food_groups"] and res2["food_groups"]["carbs"]

def test_kitchen_optimizer_staple_unlocks(sample_recipe_df):
    # User has everything for Chicken Salad except olive_oil
    # User has everything for Pasta Pomodoro except olive_oil and basil
    # olive_oil should be the top unlock!
    res = KitchenOptimizer.get_staple_unlock_report(["chicken_breast", "lettuce", "tomato", "pasta", "garlic"], sample_recipe_df)
    assert len(res) > 0
    top_unlock = res[0]["ingredient"]
    assert top_unlock == "olive oil"

def test_kitchen_optimizer_batch_cooking():
    plan = {
        "Monday": {
            "dinner": {"title": "Chicken Salad", "ingredients": "chicken|lettuce|tomato"},
        },
        "Tuesday": {
            "lunch": {"title": "Chicken Wrap", "ingredients": "chicken|tortilla|lettuce"},
        },
        "Wednesday": {
            "dinner": {"title": "Chicken Stir Fry", "ingredients": "chicken|broccoli|soy_sauce"},
        }
    }
    batch = KitchenOptimizer.find_batch_cooking_opportunities(None, plan)
    assert len(batch) >= 1
    assert batch[0]["ingredient"] == "chicken"
    assert batch[0]["used_in"] == 3

def test_recommendation_engine_facade(sample_recipe_df):
    engine = RecommendationEngine()
    
    # Test enrichment
    recipe = pd.Series({
        "title": "Tempura Veg",
        "instructions": "Deep fry in batter",
        "ingredient_list": ["carrot", "broccoli", "flour", "oil", "egg"],
        "calories": 400,
        "protein": 10,
        "coverage": 0.8,
        "missing_count": 1,
        "missing_ingredients": ["egg"]
    })
    
    enriched = engine.enrich_recipe(recipe, ["carrot", "broccoli", "flour", "oil"])
    assert "difficulty" in enriched
    assert enriched["difficulty"] == 5  # Tempura keyword
    assert "why_recommended" in enriched
    assert "1 ingredient" in enriched["why_recommended"]

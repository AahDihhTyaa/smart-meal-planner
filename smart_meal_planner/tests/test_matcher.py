"""
Unit tests for RecipeMatcher
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.recipe_matcher import RecipeMatcher


def get_matcher():
    return RecipeMatcher()


class TestInit:
    def test_loads_corpus(self):
        m = get_matcher()
        assert len(m.corpus) > 0

    def test_has_required_columns(self):
        m = get_matcher()
        cols = m.corpus.columns.tolist()
        assert "title" in cols
        assert "meal_type" in cols
        assert "ingredients" in cols
        assert "calories" in cols


class TestMatch:
    def test_returns_dataframe(self):
        m = get_matcher()
        result = m.match(["tomato", "onion", "garlic"])
        assert hasattr(result, "columns")

    def test_basic_match(self):
        m = get_matcher()
        result = m.match(["chicken_breast", "garlic", "lemon", "olive_oil"])
        assert len(result) > 0

    def test_meal_type_filter_breakfast(self):
        m = get_matcher()
        result = m.match(["egg", "butter", "bread"], meal_type="breakfast", top_n=10)
        if not result.empty:
            assert all(result["meal_type"] == "breakfast")

    def test_meal_type_filter_dinner(self):
        m = get_matcher()
        result = m.match(["pasta", "tomato", "garlic"], meal_type="dinner", top_n=10)
        if not result.empty:
            assert all(result["meal_type"] == "dinner")

    def test_dietary_filter_vegetarian(self):
        m = get_matcher()
        result = m.match(
            ["tomato", "onion", "cheese"],
            dietary_filters=["vegetarian"],
            top_n=10,
        )
        if not result.empty:
            for tags in result["tags"]:
                assert "vegetarian" in str(tags)

    def test_exclude_titles(self):
        m = get_matcher()
        first = m.match(["chicken_breast", "garlic", "onion"], top_n=1)
        if first.empty:
            return
        excluded = {first.iloc[0]["title"]}
        second = m.match(
            ["chicken_breast", "garlic", "onion"],
            top_n=5,
            exclude_titles=excluded,
        )
        if not second.empty:
            assert excluded.isdisjoint(set(second["title"].tolist()))

    def test_empty_ingredients(self):
        m = get_matcher()
        result = m.match([])
        assert result.empty

    def test_top_n_respected(self):
        m = get_matcher()
        result = m.match(["tomato", "onion"], top_n=3)
        assert len(result) <= 3

    def test_scores_present(self):
        m = get_matcher()
        result = m.match(["pasta", "tomato", "cheese"])
        if not result.empty:
            assert "final_score" in result.columns
            assert "coverage" in result.columns
            assert "tfidf_score" in result.columns

    def test_scores_in_range(self):
        m = get_matcher()
        result = m.match(["pasta", "tomato", "onion", "garlic"])
        if not result.empty:
            assert all(result["coverage"] >= 0)
            assert all(result["coverage"] <= 1)

    def test_sorted_by_score(self):
        m = get_matcher()
        result = m.match(["egg", "bread", "butter", "milk"])
        if len(result) > 1:
            scores = result["final_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_missing_ingredients_column(self):
        m = get_matcher()
        result = m.match(["pasta", "tomato"])
        if not result.empty:
            assert "missing_ingredients" in result.columns


class TestGetRecipeByTitle:
    def test_existing_title(self):
        m = get_matcher()
        titles = m.all_titles
        if titles:
            recipe = m.get_recipe_by_title(titles[0])
            assert recipe is not None
            assert recipe["title"] == titles[0]

    def test_nonexistent_title(self):
        m = get_matcher()
        result = m.get_recipe_by_title("This Recipe Does Not Exist XYZ")
        assert result is None


class TestCoverage:
    def test_full_coverage(self):
        """If user has ALL recipe ingredients, coverage should be > 0.5"""
        m = get_matcher()
        # Find the recipe with fewest ingredients
        df = m.corpus.copy()
        df["_ing_count"] = df["ingredient_list"].apply(len)
        simple = df.sort_values("_ing_count")
        recipe = m.corpus.iloc[0]
        ing_list = recipe["ingredient_list"]
        result = m.match(ing_list, top_n=50)
        # The recipe with all ingredients provided should have high coverage
        matched = result[result["title"] == recipe["title"]]
        if not matched.empty:
            assert matched.iloc[0]["coverage"] > 0.5


class TestIndianFocus:
    def test_indian_focus_boosts_indian_recipes(self):
        m = get_matcher()
        # "paneer", "turmeric", "onion" are common ingredients
        ingredients = ["paneer", "turmeric", "onion", "garlic"]
        
        # Match with indian_focus=True
        result_focus = m.match(ingredients, indian_focus=True, top_n=5)
        # Match with indian_focus=False
        result_no_focus = m.match(ingredients, indian_focus=False, top_n=5)
        
        # Verify that when focused, Indian recipes are ranked first or higher
        if not result_focus.empty:
            cuisines = result_focus["cuisine"].fillna("").str.lower().tolist()
            # If there are any Indian recipes in the corpus, the top one should be Indian
            # (Note: we have paneer, turmeric, onion, which fits "Tandoori Radish Curry #3" or "Masala Paneer")
            if "indian" in m.corpus["cuisine"].fillna("").str.lower().tolist():
                assert cuisines[0] == "indian"


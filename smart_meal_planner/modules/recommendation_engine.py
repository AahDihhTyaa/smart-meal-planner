"""
Advanced Recipe Recommendation Engine
======================================
Provides confidence analysis, difficulty estimation, kitchen optimization tips,
and seasonal availability data on top of the existing RecipeMatcher.

This module is stateless — all methods accept data and return results.
It integrates with RecipeMatcher and BudgetOptimizer for full-stack intelligence.
"""

import logging
import math
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DIFFICULTY ESTIMATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Technique keywords mapped to base difficulty
_TECHNIQUE_DIFFICULTY: Dict[str, int] = {
    # Level 5 — Pro / specialised equipment
    "tempura": 5, "soufflé": 5, "souffle": 5, "confit": 5, "flambe": 5,
    "croquembouche": 5, "choux": 5, "puff_pastry": 5, "croissant": 5,
    "sous_vide": 5, "deglaze": 5,
    # Level 4 — Advanced home cook
    "braise": 4, "braised": 4, "poach": 4, "julienne": 4, "roux": 4,
    "reduction": 4, "emulsif": 4, "caramel": 4, "risotto": 4,
    "homemade_pasta": 4, "ferment": 4, "cure": 4, "smoke": 4,
    "deep_fry": 4, "deep_fried": 4, "batter": 4,
    # Level 3 — Intermediate
    "stir_fry": 3, "stir-fry": 3, "sauté": 3, "saute": 3,
    "marinate": 3, "roast": 3, "bake": 3, "grill": 3, "bbq": 3,
    "simmer": 3, "slow_cook": 3, "blanch": 3, "fold": 3,
    "stuff": 3, "stuffed": 3, "glaze": 3, "oven": 3,
    # Level 2 — Easy
    "pan_fry": 2, "pan-fry": 2, "fry": 2, "boil": 2, "steam": 2,
    "scramble": 2, "toast": 2, "melt": 2, "mix": 2,
    # Level 1 — Trivial / no-cook
    "salad": 1, "toss": 1, "assemble": 1, "blend": 1, "smoothie": 1,
    "sandwich": 1, "wrap": 1, "raw": 1, "no_cook": 1, "overnight": 1,
}

# Technique notes for display
_TECHNIQUE_NOTES: Dict[int, str] = {
    1: "No-cook or minimal assembly",
    2: "Basic cooking skills (boil, fry, steam)",
    3: "Intermediate techniques (roasting, marinating, stir-fry)",
    4: "Advanced methods (braising, deep-frying, reductions)",
    5: "Professional techniques (tempura, soufflé, sous vide)",
}


class DifficultyEstimator:
    """Estimates recipe difficulty on a 1–5 scale from title + instructions + ingredient count."""

    @staticmethod
    def estimate(
        title: str,
        instructions: str,
        ingredient_count: int,
    ) -> Tuple[int, str]:
        """
        Returns (difficulty_level, technique_notes_string).

        Heuristics
        ----------
        1. Scan title + instructions for technique keywords → take the max.
        2. Adjust by ingredient count:  >12 ingredients → +1, ≤3 → -1.
        3. Clamp to [1, 5].
        """
        text = f"{title} {instructions}".lower().replace("-", "_")
        max_technique = 1

        for keyword, level in _TECHNIQUE_DIFFICULTY.items():
            if keyword in text:
                max_technique = max(max_technique, level)

        # Ingredient count adjustment
        if ingredient_count > 12:
            max_technique = min(max_technique + 1, 5)
        elif ingredient_count <= 3:
            max_technique = max(max_technique - 1, 1)

        level = max(1, min(5, max_technique))
        notes = _TECHNIQUE_NOTES.get(level, "")
        return level, notes

    @staticmethod
    def estimate_prep_time(ingredient_count: int, difficulty: int) -> int:
        """
        Estimate total prep+cook time in minutes.

        Heuristic: base_time = 10 + (ingredients × 3) + (difficulty × 8)
        Clamped to [5, 120].
        """
        base = 10 + (ingredient_count * 3) + (difficulty * 8)
        return max(5, min(base, 120))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEASONAL AVAILABILITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Month numbers → season name (Northern Hemisphere)
_MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

# Produce → peak seasons
_SEASONAL_PRODUCE: Dict[str, List[str]] = {
    # Vegetables
    "asparagus": ["spring"], "peas": ["spring"], "artichoke": ["spring"],
    "radish": ["spring"], "spinach": ["spring", "fall"],
    "tomato": ["summer"], "corn": ["summer"], "zucchini": ["summer"],
    "bell_pepper": ["summer"], "capsicum": ["summer"], "cucumber": ["summer"],
    "eggplant": ["summer"], "green_bean": ["summer"], "avocado": ["summer"],
    "pumpkin": ["fall"], "squash": ["fall"], "sweet_potato": ["fall"],
    "cauliflower": ["fall", "winter"], "broccoli": ["fall", "winter"],
    "kale": ["fall", "winter"], "brussels_sprout": ["fall", "winter"],
    "cabbage": ["winter"], "turnip": ["winter"], "leek": ["winter"],
    "celery": ["fall", "winter"],
    # Fruits
    "strawberry": ["spring", "summer"], "blueberry": ["summer"],
    "raspberry": ["summer"], "watermelon": ["summer"], "peach": ["summer"],
    "mango": ["summer"], "cherry": ["summer"],
    "apple": ["fall"], "pear": ["fall"], "grape": ["fall"],
    "cranberry": ["fall"], "fig": ["fall"],
    "orange": ["winter"], "grapefruit": ["winter"], "lemon": ["winter", "spring"],
    "lime": ["summer"], "pomegranate": ["fall", "winter"],
    "banana": ["spring", "summer", "fall", "winter"],  # year-round
    # Year-round staples (always in season)
    "onion": ["spring", "summer", "fall", "winter"],
    "garlic": ["spring", "summer", "fall", "winter"],
    "potato": ["spring", "summer", "fall", "winter"],
    "carrot": ["spring", "summer", "fall", "winter"],
    "mushroom": ["spring", "summer", "fall", "winter"],
}

# Seasonal swap suggestions: out-of-season → in-season alternative
_SEASONAL_SWAPS: Dict[str, Dict[str, str]] = {
    "winter": {
        "tomato": "canned_tomato", "corn": "potato",
        "zucchini": "cauliflower", "eggplant": "cabbage",
        "asparagus": "leek", "strawberry": "orange",
        "cucumber": "celery", "bell_pepper": "carrot",
    },
    "spring": {
        "pumpkin": "asparagus", "squash": "peas",
        "cranberry": "strawberry", "sweet_potato": "radish",
    },
    "summer": {
        "pumpkin": "zucchini", "kale": "spinach",
        "brussels_sprout": "green_bean", "cabbage": "lettuce",
        "orange": "mango", "apple": "peach",
    },
    "fall": {
        "watermelon": "apple", "peach": "pear",
        "asparagus": "broccoli", "blueberry": "cranberry",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AUTOMATED SUBSTITUTION MATRIX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_STANDARD_SUBSTITUTIONS: Dict[str, str] = {
    "buttermilk": "milk + 1 tbsp lemon juice",
    "soy_sauce": "tamari or coconut aminos",
    "egg": "1/4 cup applesauce or 1 flax egg",
    "heavy_cream": "milk + butter",
    "sour_cream": "plain greek yogurt",
    "baking_powder": "1/4 tsp baking soda + 1/2 tsp cream of tartar",
    "breadcrumbs": "crushed crackers or oats",
    "brown_sugar": "white sugar + molasses",
    "chicken_broth": "vegetable broth or water + bouillon",
    "lemon_juice": "lime juice or white vinegar",
}


class SeasonalAdvisor:
    """Provides seasonal availability info and swap suggestions."""

    @staticmethod
    def current_season() -> str:
        return _MONTH_TO_SEASON.get(datetime.now().month, "summer")

    @classmethod
    def is_in_season(cls, ingredient: str) -> bool:
        """Check if an ingredient is currently in peak season."""
        seasons = _SEASONAL_PRODUCE.get(ingredient.lower().strip())
        if seasons is None:
            return True  # Unknown produce assumed always available
        return cls.current_season() in seasons

    @classmethod
    def get_seasonal_swaps(cls, ingredients: List[str]) -> List[Dict[str, str]]:
        """
        For each out-of-season ingredient, suggest an in-season alternative.

        Returns list of {"original": ..., "swap": ..., "reason": ...}
        """
        season = cls.current_season()
        swaps_table = _SEASONAL_SWAPS.get(season, {})
        results = []

        for ing in ingredients:
            ing_clean = ing.lower().strip()
            if ing_clean in swaps_table:
                results.append({
                    "original": ing_clean.replace("_", " "),
                    "swap": swaps_table[ing_clean].replace("_", " "),
                    "reason": f"{ing_clean.replace('_', ' ')} is out of season in {season}; "
                              f"{swaps_table[ing_clean].replace('_', ' ')} is at peak freshness and lower cost."
                })
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIDENCE ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Food group classification for diversity scoring
_FOOD_GROUPS: Dict[str, List[str]] = {
    "protein": [
        "chicken", "beef", "lamb", "pork", "fish", "salmon", "tuna", "cod",
        "shrimp", "prawn", "tofu", "tempeh", "paneer", "egg", "turkey",
        "duck", "crab", "lobster", "lentil", "chickpea", "bean",
    ],
    "carbs": [
        "rice", "pasta", "noodle", "bread", "flour", "potato", "quinoa",
        "couscous", "oat", "barley", "corn", "sweet_potato", "tortilla",
    ],
    "vegetables": [
        "tomato", "onion", "garlic", "pepper", "carrot", "celery", "spinach",
        "broccoli", "mushroom", "zucchini", "eggplant", "peas", "kale",
        "cabbage", "cauliflower", "cucumber", "avocado", "lettuce", "leek",
        "asparagus", "green_bean", "radish", "squash", "pumpkin",
    ],
    "fats": [
        "olive_oil", "oil", "butter", "coconut_oil", "cream", "cheese",
        "avocado", "nut", "almond", "walnut", "peanut", "sesame",
    ],
    "fruits": [
        "apple", "banana", "mango", "orange", "lemon", "lime", "strawberry",
        "blueberry", "grape", "pear", "peach", "cherry", "watermelon",
    ],
}


class ConfidenceAnalyzer:
    """
    Rates how well a user's ingredient list aligns with quality recipes (0–100%).

    Scoring components:
      - Food group diversity  (40%)  — presence across protein/carbs/veg/fat/fruit
      - Ingredient count      (25%)  — more ingredients = more recipe options
      - Cuisine versatility   (20%)  — how many cuisines the ingredients span
      - Pantry completeness   (15%)  — common staples present
    """

    ESSENTIAL_STAPLES = frozenset({
        "salt", "pepper", "oil", "olive_oil", "garlic", "onion",
        "butter", "sugar", "flour", "egg",
    })

    def analyze(
        self,
        user_ingredients: List[str],
        recipe_df: pd.DataFrame,
    ) -> Dict:
        """
        Returns
        -------
        {
            "score": int (0–100),
            "grade": str ("A+" to "F"),
            "food_groups": {"protein": bool, "carbs": bool, ...},
            "missing_groups": [str],
            "staple_coverage": float,
            "tips": [str],
            "cuisine_reach": int,
        }
        """
        if not user_ingredients:
            return {
                "score": 0, "grade": "F",
                "food_groups": {}, "missing_groups": list(_FOOD_GROUPS.keys()),
                "staple_coverage": 0.0, "tips": ["Add some ingredients to get started!"],
                "cuisine_reach": 0,
            }

        user_set = set(i.lower().strip() for i in user_ingredients)

        # 1. Food group diversity (40 pts)
        group_hits = {}
        missing_groups = []
        for group, keywords in _FOOD_GROUPS.items():
            hit = any(
                kw in ing or ing in kw
                for kw in keywords for ing in user_set
            )
            group_hits[group] = hit
            if not hit:
                missing_groups.append(group)

        groups_present = sum(group_hits.values())
        diversity_score = (groups_present / len(_FOOD_GROUPS)) * 40

        # 2. Ingredient count (25 pts) — log scale, caps at 15 ingredients
        count_score = min(math.log2(max(len(user_set), 1) + 1) / math.log2(16), 1.0) * 25

        # 3. Cuisine versatility (20 pts) — how many unique cuisines match ≥30%
        cuisine_reach = 0
        if not recipe_df.empty and "cuisine" in recipe_df.columns:
            cuisines = recipe_df["cuisine"].dropna().unique()
            for cuisine in cuisines:
                cuisine_recipes = recipe_df[recipe_df["cuisine"] == cuisine]
                if len(cuisine_recipes) == 0:
                    continue
                sample = cuisine_recipes.sample(min(50, len(cuisine_recipes)), random_state=42)
                matches = 0
                for ing_list in sample["ingredient_list"]:
                    if not ing_list:
                        continue
                    covered = sum(
                        1 for i in ing_list
                        if i in user_set or any(u in i or i in u for u in user_set)
                    )
                    if covered / max(len(ing_list), 1) >= 0.3:
                        matches += 1
                if matches / max(len(sample), 1) >= 0.1:
                    cuisine_reach += 1
            cuisine_score = min(cuisine_reach / max(len(cuisines), 1), 1.0) * 20
        else:
            cuisine_score = 10  # neutral if no data

        # 4. Pantry completeness (15 pts)
        staple_hits = sum(1 for s in self.ESSENTIAL_STAPLES if s in user_set)
        staple_coverage = staple_hits / len(self.ESSENTIAL_STAPLES)
        staple_score = staple_coverage * 15

        # Total
        total = int(round(diversity_score + count_score + cuisine_score + staple_score))
        total = max(0, min(100, total))

        # Grade
        grade = self._grade(total)

        # Tips
        tips = []
        if not group_hits.get("protein"):
            tips.append("Add a protein source (chicken, tofu, eggs, lentils) to unlock hearty main dishes.")
        if not group_hits.get("carbs"):
            tips.append("Add a carb base (rice, pasta, bread, potato) for complete meals.")
        if not group_hits.get("vegetables"):
            tips.append("Add vegetables (tomato, spinach, broccoli) for nutrition and variety.")
        if staple_coverage < 0.5:
            tips.append("Stock up on pantry staples (oil, garlic, onion, salt) — they unlock hundreds of recipes.")
        if len(user_set) < 5:
            tips.append("Add more ingredients — 8+ items dramatically increases your recipe options.")

        return {
            "score": total,
            "grade": grade,
            "food_groups": group_hits,
            "missing_groups": missing_groups,
            "staple_coverage": round(staple_coverage * 100, 1),
            "tips": tips,
            "cuisine_reach": cuisine_reach,
        }

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90: return "A+"
        if score >= 80: return "A"
        if score >= 70: return "B+"
        if score >= 60: return "B"
        if score >= 50: return "C+"
        if score >= 40: return "C"
        if score >= 30: return "D"
        return "F"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KITCHEN OPTIMIZER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KitchenOptimizer:
    """
    Provides actionable kitchen intelligence:
    - Missing staple ingredients that unlock the most new recipes
    - Batch-cooking opportunities across the meal plan
    - Seasonal swap suggestions
    """

    @staticmethod
    def get_staple_unlock_report(
        user_ingredients: List[str],
        recipe_df: pd.DataFrame,
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Find single ingredients that, if added, would unlock the most new recipes.

        Returns list of {"ingredient": str, "unlocks": int, "top_recipes": [str]}
        """
        if recipe_df.empty:
            return []

        user_set = set(i.lower().strip() for i in user_ingredients)
        # Expand user set for matching
        expanded = set(user_set)
        for ing in user_set:
            parts = ing.split("_")
            for p in parts:
                if len(p) > 3:
                    expanded.add(p)

        # Count how often each non-user ingredient appears in "almost matched" recipes
        # (recipes with coverage ≥ 60% but missing 1–2 ingredients)
        candidate_counter: Counter = Counter()
        candidate_recipes: Dict[str, List[str]] = {}

        for _, row in recipe_df.iterrows():
            ing_list = row.get("ingredient_list", [])
            if not ing_list:
                continue
            have = []
            missing = []
            for i in ing_list:
                i_base = i.split("_")[0]
                if (i in expanded or i_base in expanded
                        or any(u in i for u in user_set)
                        or any(i in u for u in user_set)):
                    have.append(i)
                else:
                    missing.append(i)

            coverage = len(have) / max(len(ing_list), 1)
            if coverage >= 0.55 and 1 <= len(missing) <= 2:
                for m in missing:
                    candidate_counter[m] += 1
                    if m not in candidate_recipes:
                        candidate_recipes[m] = []
                    if len(candidate_recipes[m]) < 3:
                        candidate_recipes[m].append(row.get("title", "Unknown"))

        results = []
        for ing, count in candidate_counter.most_common(top_n):
            results.append({
                "ingredient": ing.replace("_", " "),
                "unlocks": count,
                "top_recipes": candidate_recipes.get(ing, []),
            })
        return results

    @staticmethod
    def find_batch_cooking_opportunities(
        day_plan: Dict,
        week_plan: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Find ingredients that appear in 3+ recipes across the plan.
        These are batch-cooking candidates (prep once, use multiple times).

        Returns list of {"ingredient": str, "used_in": int, "meals": [str]}
        """
        ingredient_usage: Dict[str, List[str]] = {}
        plans_to_check = {}

        if week_plan:
            for day, dp in week_plan.items():
                for slot, recipe in dp.items():
                    if recipe is not None:
                        plans_to_check[f"{day} {slot}"] = recipe
        elif day_plan:
            for slot, recipe in day_plan.items():
                if recipe is not None:
                    plans_to_check[slot] = recipe

        for meal_label, recipe in plans_to_check.items():
            ing_list = recipe.get("ingredient_list")
            if not ing_list:
                ing_str = str(recipe.get("ingredients", ""))
                ing_list = [i.strip() for i in ing_str.split("|") if i.strip()]
            title = recipe.get("title", meal_label)
            for ing in ing_list:
                clean = ing.replace("_", " ")
                if clean not in ingredient_usage:
                    ingredient_usage[clean] = []
                ingredient_usage[clean].append(title)

        results = []
        for ing, meals in sorted(ingredient_usage.items(), key=lambda x: -len(x[1])):
            if len(meals) >= 3:
                results.append({
                    "ingredient": ing,
                    "used_in": len(meals),
                    "meals": list(set(meals))[:5],
                })
        return results

    @staticmethod
    def generate_why_this_recipe(
        recipe: pd.Series,
        user_ingredients: List[str],
    ) -> str:
        """
        Generate a human-readable explanation of why this recipe was recommended.
        """
        coverage = float(recipe.get("coverage", 0)) * 100
        missing_count = int(recipe.get("missing_count", 0))
        missing = recipe.get("missing_ingredients", [])
        title = recipe.get("title", "This recipe")
        cuisine = recipe.get("cuisine", "")

        reasons = []

        if missing_count == 0:
            reasons.append(f"You have **every ingredient** needed for {title} — no shopping required!")
        elif missing_count <= 2:
            reasons.append(f"You're only missing **{missing_count} ingredient{'s' if missing_count > 1 else ''}** "
                          f"({', '.join(m.replace('_', ' ') for m in missing[:3])}).")
        else:
            reasons.append(f"**{coverage:.0f}%** of ingredients are in your pantry.")

        if cuisine:
            reasons.append(f"Brings **{cuisine.capitalize()}** cuisine variety to your plan.")

        # Nutritional note
        cals = int(recipe.get("calories", 0))
        protein = float(recipe.get("protein", 0))
        if protein > 30:
            reasons.append(f"High-protein meal with **{protein:.0f}g protein** per serving.")
        if 300 <= cals <= 500:
            reasons.append(f"Well-balanced at **{cals} kcal** per serving.")

        return " ".join(reasons)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN ENGINE (FACADE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RecommendationEngine:
    """
    Facade class that aggregates all recommendation intelligence.
    Stateless — pass in data, get back enriched results.
    """

    def __init__(self):
        self.confidence = ConfidenceAnalyzer()
        self.difficulty = DifficultyEstimator()
        self.kitchen = KitchenOptimizer()
        self.seasonal = SeasonalAdvisor()

    def enrich_recipe(self, recipe: pd.Series, user_ingredients: List[str]) -> Dict:
        """
        Add difficulty, prep time, technique notes, and 'why it works'
        to a single recipe result.

        Returns a dict with the extra fields.
        """
        title = str(recipe.get("title", ""))
        instructions = str(recipe.get("instructions", ""))
        ing_list = recipe.get("ingredient_list", [])
        ing_count = len(ing_list) if ing_list else 0

        diff_level, diff_notes = self.difficulty.estimate(title, instructions, ing_count)
        prep_time = self.difficulty.estimate_prep_time(ing_count, diff_level)
        why = self.kitchen.generate_why_this_recipe(recipe, user_ingredients)

        return {
            "difficulty": diff_level,
            "difficulty_notes": diff_notes,
            "prep_time_min": prep_time,
            "why_recommended": why,
        }

    def get_full_analysis(
        self,
        user_ingredients: List[str],
        recipe_df: pd.DataFrame,
        day_plan: Optional[Dict] = None,
        week_plan: Optional[Dict] = None,
    ) -> Dict:
        """
        Run all analyses and return a comprehensive report.

        Returns
        -------
        {
            "confidence": {...},
            "staple_unlocks": [...],
            "batch_cooking": [...],
            "seasonal_swaps": [...],
            "current_season": str,
        }
        """
        confidence = self.confidence.analyze(user_ingredients, recipe_df)
        staples = self.kitchen.get_staple_unlock_report(user_ingredients, recipe_df)
        batch = self.kitchen.find_batch_cooking_opportunities(
            day_plan or {}, week_plan
        )
        seasonal = self.seasonal.get_seasonal_swaps(user_ingredients)
        season = self.seasonal.current_season()

        return {
            "confidence": confidence,
            "staple_unlocks": staples,
            "batch_cooking": batch,
            "seasonal_swaps": seasonal,
            "current_season": season,
        }

    def get_substitutions(self, missing_ingredients: List[str]) -> List[Dict]:
        """
        Looks up missing ingredients in the standard substitution matrix.
        """
        subs = []
        for ing in missing_ingredients:
            clean = ing.replace('_', ' ').lower().strip()
            if clean in _STANDARD_SUBSTITUTIONS:
                subs.append({"ingredient": clean, "substitute": _STANDARD_SUBSTITUTIONS[clean]})
            elif ing in _STANDARD_SUBSTITUTIONS:
                subs.append({"ingredient": clean, "substitute": _STANDARD_SUBSTITUTIONS[ing]})
        return subs

"""
Budget Optimizer Module
=======================
Estimates food costs per ingredient, recipe, and meal plan.
Provides budget alerts and low-cost ingredient swap recommendations.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


# ── Cost database (USD per typical serving / unit) ─────────────────────────────
# Categories assign a base cost per serving (~80-200g)
_INGREDIENT_COSTS: Dict[str, float] = {
    # Proteins — expensive
    "salmon": 3.50, "tuna": 2.80, "shrimp": 3.20, "prawn": 3.20,
    "beef": 3.00, "lamb": 3.50, "steak": 4.00,
    "chicken": 2.00, "chicken_breast": 2.20, "turkey": 2.50,
    "pork": 1.80, "bacon": 2.00,
    "cod": 2.80, "fish": 2.60,
    # Plant proteins — moderate
    "tofu": 1.00, "tempeh": 1.20, "paneer": 1.50,
    "egg": 0.30, "eggs": 0.30,
    "lentil": 0.40, "lentils": 0.40,
    "chickpea": 0.45, "chickpeas": 0.45,
    "bean": 0.40, "beans": 0.40,
    "black_bean": 0.40, "kidney_bean": 0.40,
    # Dairy
    "milk": 0.25, "cheese": 0.80, "parmesan": 1.20, "cheddar": 0.90,
    "mozzarella": 1.00, "feta": 1.00, "yogurt": 0.50, "cream": 0.60,
    "butter": 0.30,
    # Grains / starch
    "rice": 0.20, "pasta": 0.25, "noodle": 0.25, "noodles": 0.25,
    "bread": 0.30, "flour": 0.15, "quinoa": 0.60,
    "oat": 0.20, "oats": 0.20, "couscous": 0.30, "barley": 0.25,
    # Vegetables — cheap
    "tomato": 0.40, "onion": 0.20, "garlic": 0.10, "ginger": 0.15,
    "potato": 0.25, "sweet_potato": 0.35, "carrot": 0.20, "celery": 0.20,
    "spinach": 0.40, "kale": 0.45, "lettuce": 0.30, "cabbage": 0.20,
    "broccoli": 0.50, "cauliflower": 0.55, "corn": 0.30,
    "mushroom": 0.60, "bell_pepper": 0.50, "capsicum": 0.50,
    "zucchini": 0.40, "eggplant": 0.45, "cucumber": 0.30,
    "avocado": 0.80, "peas": 0.30, "green_bean": 0.40,
    # Fruits
    "lemon": 0.25, "lime": 0.20, "apple": 0.35, "banana": 0.20,
    "mango": 0.60, "orange": 0.35, "strawberry": 0.70,
    # Pantry / oils / sauces — very cheap
    "olive_oil": 0.15, "oil": 0.10, "vegetable_oil": 0.08,
    "soy_sauce": 0.08, "tomato_paste": 0.10, "vinegar": 0.05,
    "salt": 0.01, "pepper": 0.02, "sugar": 0.05,
    "cumin": 0.05, "turmeric": 0.05, "paprika": 0.05,
    "coriander": 0.05, "garam_masala": 0.08, "curry": 0.07,
    "coconut_milk": 0.60, "broth": 0.30, "stock": 0.30,
    "honey": 0.20, "maple_syrup": 0.40,
}

# Fallback cost categories
_CATEGORY_COSTS = {
    "meat_fish": 2.80,
    "plant_protein": 0.80,
    "dairy": 0.60,
    "grain": 0.25,
    "vegetable": 0.35,
    "pantry": 0.10,
}

# Cheap swap suggestions — maps expensive → cheap alternative
_SWAP_SUGGESTIONS: Dict[str, Tuple[str, float]] = {
    "salmon": ("tilapia", 1.50),
    "beef": ("chicken", 2.00),
    "lamb": ("pork", 1.80),
    "shrimp": ("tofu", 1.00),
    "prawn": ("tofu", 1.00),
    "steak": ("chicken_breast", 2.20),
    "parmesan": ("cheddar", 0.90),
    "mozzarella": ("ricotta", 0.70),
    "quinoa": ("rice", 0.20),
    "avocado": ("cucumber", 0.30),
}


class BudgetOptimizer:
    """Estimates ingredient & meal costs; suggests swaps to meet budget."""

    def __init__(self, daily_budget: float = 15.0):
        self.daily_budget = daily_budget

    # ── Cost estimation ────────────────────────────────────────────────────────

    def ingredient_cost(self, ingredient: str) -> float:
        """Return estimated cost (USD) per serving for an ingredient."""
        name = ingredient.lower().strip()
        if name in _INGREDIENT_COSTS:
            return _INGREDIENT_COSTS[name]
        # Try partial match — prefer longest matching key to avoid false positives
        best_match = None
        best_len = 0
        for key, cost in _INGREDIENT_COSTS.items():
            if key in name or name in key:
                if len(key) > best_len:
                    best_match = cost
                    best_len = len(key)
        if best_match is not None:
            return best_match
        return _CATEGORY_COSTS["pantry"]  # unknown → assume pantry

    def recipe_cost(self, recipe: pd.Series) -> float:
        """Sum ingredient costs for a single recipe row."""
        ingredients = str(recipe.get("ingredients", ""))
        ing_list = [i.strip() for i in ingredients.split("|") if i.strip()]
        total = sum(self.ingredient_cost(ing) for ing in ing_list)
        return round(total, 2)

    def plan_cost(self, day_plan: Dict) -> Dict[str, float]:
        """Return per-slot and total cost for a day plan."""
        costs = {}
        total = 0.0
        for slot, recipe in day_plan.items():
            if recipe is not None:
                c = self.recipe_cost(recipe)
                costs[slot] = c
                total += c
        costs["total"] = round(total, 2)
        return costs

    def shopping_list_cost(self, missing_items: List[str]) -> float:
        """Estimate total shopping cost for a list of missing ingredients."""
        total = sum(self.ingredient_cost(item) for item in missing_items)
        return round(total, 2)

    # ── Budget alerts & swaps ─────────────────────────────────────────────────

    def get_budget_feedback(
        self,
        daily_cost: float,
        recipe_costs: Dict[str, float],
        day_plan: Dict,
    ) -> Dict:
        """
        Return budget status, alerts and swap suggestions for the daily plan.
        """
        over_budget = daily_cost > self.daily_budget
        pct = (daily_cost / self.daily_budget * 100) if self.daily_budget > 0 else 0
        alerts = []
        swaps = []

        if over_budget:
            alerts.append({
                "type": "warning",
                "message": f"💸 Over Budget: Daily meal cost is ${daily_cost:.2f}, exceeding your ${self.daily_budget:.2f}/day budget by ${daily_cost - self.daily_budget:.2f}."
            })
            # Identify most expensive recipe
            sorted_slots = sorted(recipe_costs.items(), key=lambda x: x[1], reverse=True)
            for slot, cost in sorted_slots:
                recipe = day_plan.get(slot)
                if recipe is None:
                    continue
                ing_list = [i.strip() for i in str(recipe.get("ingredients", "")).split("|") if i.strip()]
                for ing in ing_list:
                    base = ing.split("_")[0]
                    if base in _SWAP_SUGGESTIONS:
                        alt, alt_cost = _SWAP_SUGGESTIONS[base]
                        orig_cost = self.ingredient_cost(ing)
                        saving = round(orig_cost - alt_cost, 2)
                        if saving > 0:
                            swaps.append({
                                "slot": slot,
                                "original": ing.replace("_", " "),
                                "swap": alt.replace("_", " "),
                                "saving": saving
                            })
                if swaps:
                    break
            if not swaps:
                alerts.append({
                    "type": "info",
                    "message": "💡 Tip: Try switching to plant-based proteins (tofu, lentils, beans) to reduce meal costs significantly."
                })
        else:
            alerts.append({
                "type": "success",
                "message": f"✅ Within Budget: Daily meal cost is ${daily_cost:.2f} — ${self.daily_budget - daily_cost:.2f} under your ${self.daily_budget:.2f}/day limit."
            })

        return {
            "over_budget": over_budget,
            "pct_used": round(pct, 1),
            "daily_cost": daily_cost,
            "alerts": alerts,
            "swaps": swaps,
        }

    def weekly_cost(self, week_plan: Dict) -> Dict[str, float]:
        """Return per-day and total cost for a weekly plan."""
        result = {}
        grand_total = 0.0
        for day, day_plan in week_plan.items():
            costs = self.plan_cost(day_plan)
            result[day] = costs.get("total", 0.0)
            grand_total += result[day]
        result["TOTAL"] = round(grand_total, 2)
        return result

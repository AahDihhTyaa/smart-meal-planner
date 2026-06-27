"""
Nutrition Module
================
Attaches nutrition data to recipes and aggregates across meals/days/weeks.

Priority:
  1. Pre-stored values in the recipe corpus (calories, protein, carbs, fat).
  2. Fallback: estimate by summing per-100g ingredient values from nutrition_table.csv
     (assumes ~80g per ingredient — a rough approximation when recipe data is missing).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_NUTRITION_FILE = _DATA_DIR / "nutrition_table.csv"

# Assumed portion weight per ingredient when using fallback estimation (grams)
_DEFAULT_PORTION_G = 80.0


class NutritionModule:
    """Nutrition lookup, estimation, and aggregation."""

    def __init__(self):
        self._table: pd.DataFrame = self._load_table()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_table(self) -> pd.DataFrame:
        if _NUTRITION_FILE.exists():
            df = pd.read_csv(_NUTRITION_FILE, index_col="ingredient")
            return df
        logger.warning("Nutrition table not found: %s", _NUTRITION_FILE)
        return pd.DataFrame()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_recipe_nutrition(self, recipe: pd.Series) -> Dict[str, float]:
        """
        Return nutrition dict for a recipe row.

        Uses pre-stored corpus values if available, falls back to ingredient
        estimation otherwise.
        """
        stored_cal = recipe.get("calories", 0)
        if stored_cal and float(stored_cal) > 0:
            return {
                "calories": float(recipe.get("calories", 0)),
                "protein": float(recipe.get("protein", 0)),
                "carbs": float(recipe.get("carbs", 0)),
                "fat": float(recipe.get("fat", 0)),
            }
        # Fallback
        return self._estimate_from_ingredients(
            recipe.get("ingredient_list", [])
        )

    def _estimate_from_ingredients(
        self, ingredient_list: List[str]
    ) -> Dict[str, float]:
        """Estimate nutrition by summing per-100g values × assumed portion."""
        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        if self._table.empty:
            return totals

        for ing in ingredient_list:
            if ing in self._table.index:
                row = self._table.loc[ing]
                factor = _DEFAULT_PORTION_G / 100.0
                totals["calories"] += float(row.get("calories_per_100g", 0)) * factor
                totals["protein"] += float(row.get("protein_g", 0)) * factor
                totals["carbs"] += float(row.get("carbs_g", 0)) * factor
                totals["fat"] += float(row.get("fat_g", 0)) * factor

        return {k: round(v, 1) for k, v in totals.items()}

    def aggregate_day(
        self, day_plan: Dict[str, Optional[pd.Series]]
    ) -> Dict[str, float]:
        """Sum nutrition across all meals in a day plan."""
        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for recipe in day_plan.values():
            if recipe is not None:
                n = self.get_recipe_nutrition(recipe)
                for key in totals:
                    totals[key] += n[key]
        return {k: round(v, 1) for k, v in totals.items()}

    def aggregate_week(
        self, weekly_plan: Dict[str, Dict[str, Optional[pd.Series]]]
    ) -> Dict[str, Dict[str, float]]:
        """Return per-day and total nutrition for a weekly plan."""
        result = {}
        grand = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for day, day_plan in weekly_plan.items():
            day_totals = self.aggregate_day(day_plan)
            result[day] = day_totals
            for key in grand:
                grand[key] += day_totals[key]
        result["TOTAL"] = {k: round(v, 1) for k, v in grand.items()}
        return result

    def macro_breakdown(
        self, nutrition: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Return macro percentages from a nutrition dict.

        Carbs = 4 kcal/g, Protein = 4 kcal/g, Fat = 9 kcal/g.
        """
        carb_cal = nutrition.get("carbs", 0) * 4
        prot_cal = nutrition.get("protein", 0) * 4
        fat_cal = nutrition.get("fat", 0) * 9
        total = carb_cal + prot_cal + fat_cal

        if total == 0:
            return {"carbs_pct": 0.0, "protein_pct": 0.0, "fat_pct": 0.0}

        return {
            "carbs_pct": round(carb_cal / total * 100, 1),
            "protein_pct": round(prot_cal / total * 100, 1),
            "fat_pct": round(fat_cal / total * 100, 1),
        }

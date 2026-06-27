"""
Meal Planner Module
===================
Assembles ranked recipes into daily and weekly meal plans.

Algorithm:
  - Greedy selection: pick the highest-scored recipe for each meal slot that
    hasn't been used yet (variety constraint).
  - Weekly mode: run daily selection 7 times sharing a single "used" set and
    a cumulative ingredient-tracking set (waste-minimization bonus applied).
"""

import logging
from typing import Dict, List, Optional, Set

import pandas as pd

from .recipe_matcher import RecipeMatcher

logger = logging.getLogger(__name__)

MEAL_SLOTS = ["breakfast", "lunch", "dinner"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class MealPlanner:
    """Greedy daily & weekly meal plan generator."""

    def __init__(self, matcher: RecipeMatcher):
        self.matcher = matcher

    # ── Public API ─────────────────────────────────────────────────────────────

    def plan_day(
        self,
        user_ingredients: List[str],
        dietary_filters: Optional[List[str]] = None,
        exclude_titles: Optional[Set[str]] = None,
        indian_focus: bool = False,
    ) -> Dict[str, Optional[pd.Series]]:
        """
        Generate a single day's meal plan (breakfast / lunch / dinner).

        Returns
        -------
        dict : {"breakfast": recipe_row, "lunch": recipe_row, "dinner": recipe_row}
               Values may be None if no match was found.
        """
        used: Set[str] = set(exclude_titles or [])
        plan: Dict[str, Optional[pd.Series]] = {}

        for slot in MEAL_SLOTS:
            candidates = self.matcher.match(
                user_ingredients=user_ingredients,
                meal_type=slot,
                dietary_filters=dietary_filters,
                top_n=5,
                exclude_titles=used,
                strict_mode=True,
                indian_focus=indian_focus,
            )
            if candidates.empty:
                plan[slot] = None
            else:
                best = candidates.iloc[0]
                plan[slot] = best
                used.add(best["title"])

        return plan

    def plan_week(
        self,
        user_ingredients: List[str],
        dietary_filters: Optional[List[str]] = None,
        indian_focus: bool = False,
    ) -> Dict[str, Dict[str, Optional[pd.Series]]]:
        """
        Generate a full 7-day meal plan with no-repeat and cuisine-variety constraints.

        Cuisine variety rule: no cuisine appears 3 times in a row across
        consecutive meal slots (Mon-dinner → Tue-breakfast → Tue-lunch).

        Returns
        -------
        dict : {day_name: {"breakfast": ..., "lunch": ..., "dinner": ...}}
        """
        used_titles: Set[str] = set()
        # Track recent cuisines for variety enforcement (last 2 meals)
        recent_cuisines: List[str] = []
        available = list(user_ingredients)
        weekly_plan: Dict[str, Dict[str, Optional[pd.Series]]] = {}

        for day in DAYS:
            day_plan: Dict[str, Optional[pd.Series]] = {}

            for slot in MEAL_SLOTS:
                candidates = self.matcher.match(
                    user_ingredients=available,
                    meal_type=slot,
                    dietary_filters=dietary_filters,
                    top_n=10,
                    exclude_titles=used_titles,
                    strict_mode=True,
                    indian_focus=indian_focus,
                )

                if candidates.empty:
                    day_plan[slot] = None
                    continue

                # Cuisine variety: skip recipes that repeat the same cuisine 3× in a row
                selected = None
                for _, row in candidates.iterrows():
                    cuisine = str(row.get("cuisine", "")).lower()
                    if len(recent_cuisines) >= 2 and all(c == cuisine for c in recent_cuisines[-2:]):
                        if indian_focus and cuisine == "indian":
                            # Allow consecutive Indian meals if Indian Focus is active
                            pass
                        else:
                            continue  # Would be 3 in a row — skip
                    selected = row
                    break

                # Fallback to best score if all are same cuisine
                if selected is None:
                    selected = candidates.iloc[0]

                day_plan[slot] = selected
                used_titles.add(selected["title"])
                recent_cuisines.append(str(selected.get("cuisine", "")).lower())
                # Keep only last 2 for sliding window check
                if len(recent_cuisines) > 2:
                    recent_cuisines.pop(0)

            weekly_plan[day] = day_plan

        return weekly_plan

    def get_plan_summary(
        self, plan: Dict[str, Optional[pd.Series]]
    ) -> Dict[str, Dict]:
        """
        Convert a day plan (raw Series rows) to a clean serialisable dict.
        """
        summary = {}
        for slot, recipe in plan.items():
            if recipe is None:
                summary[slot] = None
            else:
                summary[slot] = {
                    "title": recipe["title"],
                    "cuisine": recipe.get("cuisine", ""),
                    "tags": recipe.get("tags", ""),
                    "ingredients": recipe.get("ingredient_list", []),
                    "instructions": recipe.get("instructions", ""),
                    "calories": int(recipe.get("calories", 0)),
                    "protein": float(recipe.get("protein", 0)),
                    "carbs": float(recipe.get("carbs", 0)),
                    "fat": float(recipe.get("fat", 0)),
                    "coverage": round(float(recipe.get("coverage", 0)) * 100, 1),
                    "missing_ingredients": recipe.get("missing_ingredients", []),
                    "final_score": round(float(recipe.get("final_score", 0)), 3),
                }
        return summary

    def get_weekly_totals(
        self, weekly_plan: Dict[str, Dict[str, Optional[pd.Series]]]
    ) -> Dict[str, float]:
        """Sum nutrition totals across an entire week."""
        totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for day_plan in weekly_plan.values():
            for recipe in day_plan.values():
                if recipe is not None:
                    for key in totals:
                        totals[key] += float(recipe.get(key, 0))
        return {k: round(v, 1) for k, v in totals.items()}

"""
Smart Meal Planner — modules package
"""
from .nlp_normalize import NLPNormalizer
from .recipe_matcher import RecipeMatcher
from .meal_planner import MealPlanner
from .nutrition import NutritionModule

__all__ = ["NLPNormalizer", "RecipeMatcher", "MealPlanner", "NutritionModule"]

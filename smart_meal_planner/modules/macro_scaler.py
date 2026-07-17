import copy
from typing import Dict
import pandas as pd

class MacroScaler:
    """
    Dynamic Workout & Activity Macro Scaler.
    Scales macros and ingredients based on daily activity level.
    """
    
    # Define scaling factors for Carbs and Protein based on activity
    ACTIVITY_MULTIPLIERS = {
        "sedentary": {"protein": 1.0, "carbs": 1.0},
        "moderate": {"protein": 1.15, "carbs": 1.2},
        "active": {"protein": 1.3, "carbs": 1.4},
        "intense": {"protein": 1.5, "carbs": 1.6}
    }

    @staticmethod
    def scale_recipe(recipe: Dict, activity_level: str) -> Dict:
        """
        Scale a single recipe's macros and ingredient lists.
        """
        level = activity_level.lower().strip()
        multipliers = MacroScaler.ACTIVITY_MULTIPLIERS.get(level, MacroScaler.ACTIVITY_MULTIPLIERS["sedentary"])
        
        if not recipe:
            return recipe
            
        scaled_recipe = copy.deepcopy(recipe)
        
        p_mult = multipliers["protein"]
        c_mult = multipliers["carbs"]
        
        # Assuming fats scale slightly with protein
        f_mult = 1.0 + ((p_mult - 1.0) / 2)
        
        # Scale Nutritional values
        orig_protein = float(scaled_recipe.get("protein", 0))
        orig_carbs = float(scaled_recipe.get("carbs", 0))
        orig_fat = float(scaled_recipe.get("fat", 0))
        orig_calories = float(scaled_recipe.get("calories", 0))
        
        scaled_recipe["protein"] = round(orig_protein * p_mult, 1)
        scaled_recipe["carbs"] = round(orig_carbs * c_mult, 1)
        scaled_recipe["fat"] = round(orig_fat * f_mult, 1)
        
        # Recalculate calories (4 kcal/g for protein/carbs, 9 kcal/g for fat)
        scaled_recipe["calories"] = round(
            (scaled_recipe["protein"] * 4) + 
            (scaled_recipe["carbs"] * 4) + 
            (scaled_recipe["fat"] * 9)
        )
        
        # We optionally add a badge/note to the recipe indicating it was scaled
        scaled_recipe["scaled_note"] = f"Scaled for {level} activity (+{int((p_mult-1)*100)}% Protein, +{int((c_mult-1)*100)}% Carbs)"
        
        return scaled_recipe

    @staticmethod
    def scale_day_plan(day_plan: Dict, activity_level: str) -> Dict:
        """
        Scale an entire day's meal plan based on activity level.
        """
        scaled_plan = {}
        for slot, recipe in day_plan.items():
            if recipe is not None:
                scaled_plan[slot] = MacroScaler.scale_recipe(recipe, activity_level)
            else:
                scaled_plan[slot] = None
        return scaled_plan

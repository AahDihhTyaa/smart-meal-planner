from typing import Dict, List
from .budget_optimizer import BudgetOptimizer

class PersonalAnalytics:
    """
    Personal Analytics & Sustainability Dashboard Engine.
    Computes metrics for budget tracking, macro distribution, and waste reduction.
    """
    
    def __init__(self, budget_optimizer: BudgetOptimizer):
        self.budget_optimizer = budget_optimizer
        
    def calculate_macro_distribution(self, week_plan: Dict) -> Dict:
        """Calculates total weekly macros for charting."""
        totals = {"protein": 0.0, "carbs": 0.0, "fat": 0.0}
        for day, day_plan in week_plan.items():
            for slot, recipe in day_plan.items():
                if recipe:
                    totals["protein"] += float(recipe.get("protein", 0))
                    totals["carbs"] += float(recipe.get("carbs", 0))
                    totals["fat"] += float(recipe.get("fat", 0))
        
        return {
            "labels": ["Protein", "Carbs", "Fat"],
            "values": [round(totals["protein"], 1), round(totals["carbs"], 1), round(totals["fat"], 1)]
        }
        
    def calculate_budget_tracking(self, week_plan: Dict) -> Dict:
        """Calculates daily spend vs budget for the week."""
        days = list(week_plan.keys())
        daily_costs = []
        budget_line = []
        
        for day in days:
            cost_info = self.budget_optimizer.plan_cost(week_plan[day])
            daily_costs.append(cost_info.get("total", 0.0))
            budget_line.append(self.budget_optimizer.daily_budget)
            
        total_spend = sum(daily_costs)
        total_budget = self.budget_optimizer.daily_budget * len(days)
            
        return {
            "days": days,
            "daily_costs": daily_costs,
            "budget_line": budget_line,
            "total_spend": round(total_spend, 2),
            "total_budget": round(total_budget, 2)
        }
        
    def calculate_sustainability_score(self, week_plan: Dict) -> Dict:
        """
        Mock Sustainability/Waste Score based on ingredient reuse.
        Higher overlap of ingredients across the week = less waste = higher score.
        """
        all_ingredients = []
        unique_ingredients = set()
        
        for day_plan in week_plan.values():
            for recipe in day_plan.values():
                if recipe:
                    # Depending on how ingredients are stored, we handle lists or strings
                    ings = recipe.get("ingredients", [])
                    if isinstance(ings, str):
                        ings = [i.strip() for i in ings.split("|") if i.strip()]
                    
                    for ing in ings:
                        # simple normalization
                        cleaned = ing.lower().split('_')[0] 
                        all_ingredients.append(cleaned)
                        unique_ingredients.add(cleaned)
                        
        total_uses = len(all_ingredients)
        unique_count = len(unique_ingredients)
        
        if total_uses == 0:
            return {"score": 0, "message": "No data"}
            
        # If we use 30 ingredients total, and only 10 unique, reuse ratio is 3.0
        # Let's map a reuse ratio to a 0-100 score.
        reuse_ratio = total_uses / max(unique_count, 1)
        
        # Assuming a ratio of 1.0 is 50%, and 3.0+ is 95%+
        score = min(max((reuse_ratio - 1) * 25 + 50, 0), 100)
        
        return {
            "score": int(score),
            "unique_ingredients": unique_count,
            "total_ingredient_uses": total_uses,
            "message": f"You reused ingredients an average of {round(reuse_ratio, 1)} times this week!"
        }
        
    def generate_full_report(self, week_plan: Dict) -> Dict:
        """Generate the complete analytics payload for the frontend."""
        if not week_plan:
            return {"error": "No week plan provided."}
            
        return {
            "macros": self.calculate_macro_distribution(week_plan),
            "budget": self.calculate_budget_tracking(week_plan),
            "sustainability": self.calculate_sustainability_score(week_plan)
        }

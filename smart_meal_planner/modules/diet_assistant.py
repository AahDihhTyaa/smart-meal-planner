"""
Diet & Nutrition Assistant Module
=================================
Calculates metabolic indicators (BMR, TDEE) and target macronutrient splits
based on user health goals, physical statistics, and activity profiles.
Also handles clinical-dietitian prompt engineering for the AI Diet Coach.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DietAssistantModule:
    """Calculates nutritional targets and generates system prompts for the AI Dietitian."""

    ACTIVITY_MULTIPLIERS = {
        "Sedentary (Little/no exercise)": 1.2,
        "Lightly Active (Exercise 1-3 days/week)": 1.375,
        "Moderately Active (Exercise 3-5 days/week)": 1.55,
        "Very Active (Hard exercise 6-7 days/week)": 1.725,
        "Extra Active (Hard physical job & sports)": 1.9
    }

    GOAL_MULTIPLIERS = {
        "Maintain Weight": 1.0,
        "Weight Loss (Moderate deficit)": 0.85,
        "Weight Loss (Aggressive deficit)": 0.75,
        "Weight Gain (Mild surplus)": 1.10,
        "Muscle Gain (Clean bulk)": 1.15
    }

    MACRO_PRESETS = {
        "Balanced (50/25/25)": {"carbs": 50, "protein": 25, "fat": 25},
        "High Protein (40/35/25)": {"carbs": 40, "protein": 35, "fat": 25},
        "Low Carb (25/35/40)": {"carbs": 25, "protein": 35, "fat": 40},
        "Keto (5/25/70)": {"carbs": 5, "protein": 25, "fat": 70},
        "Mediterranean (45/20/35)": {"carbs": 45, "protein": 20, "fat": 35}
    }

    @staticmethod
    def calculate_bmr(weight_kg: float, height_cm: float, age_years: int, gender: str) -> float:
        """
        Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation.
        """
        if weight_kg <= 0 or height_cm <= 0 or age_years <= 0:
            return 0.0

        # Mifflin-St Jeor Formula
        if gender.lower() == "male":
            return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years) + 5.0
        else:
            return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age_years) - 161.0

    @classmethod
    def calculate_tdee(cls, bmr: float, activity_level: str) -> float:
        """Calculate Total Daily Energy Expenditure (TDEE) based on activity level."""
        multiplier = cls.ACTIVITY_MULTIPLIERS.get(activity_level, 1.2)
        return round(bmr * multiplier, 1)

    @classmethod
    def calculate_targets(
        cls,
        weight_kg: float,
        height_cm: float,
        age_years: int,
        gender: str,
        activity_level: str,
        diet_goal: str,
        macro_preset: str
    ) -> Dict[str, Any]:
        """
        Calculate complete targets including target calories and macro weights in grams.
        - Carbs: 4 kcal/g
        - Protein: 4 kcal/g
        - Fat: 9 kcal/g
        """
        bmr = cls.calculate_bmr(weight_kg, height_cm, age_years, gender)
        if bmr == 0.0:
            return {
                "bmr": 0.0,
                "tdee": 0.0,
                "target_calories": 2000.0,
                "macro_targets": {"carbs": 250.0, "protein": 125.0, "fat": 56.0},
                "macro_percentages": {"carbs": 50, "protein": 25, "fat": 25}
            }

        tdee = cls.calculate_tdee(bmr, activity_level)
        goal_mult = cls.GOAL_MULTIPLIERS.get(diet_goal, 1.0)
        target_calories = round(tdee * goal_mult, 0)

        # Retrieve macro percentage split
        percentages = cls.MACRO_PRESETS.get(macro_preset, {"carbs": 50, "protein": 25, "fat": 25})

        # Calculate grams based on caloric density
        carb_calories = target_calories * (percentages["carbs"] / 100.0)
        protein_calories = target_calories * (percentages["protein"] / 100.0)
        fat_calories = target_calories * (percentages["fat"] / 100.0)

        target_carbs_g = round(carb_calories / 4.0, 1)
        target_protein_g = round(protein_calories / 4.0, 1)
        target_fat_g = round(fat_calories / 9.0, 1)

        return {
            "bmr": round(bmr, 1),
            "tdee": tdee,
            "target_calories": target_calories,
            "macro_targets": {
                "carbs": target_carbs_g,
                "protein": target_protein_g,
                "fat": target_fat_g
            },
            "macro_percentages": percentages
        }

    @staticmethod
    def get_compliance_feedback(actual: Dict[str, float], targets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares actual nutrition against target profiles and yields warnings or successes.
        """
        feedback = []
        is_compliant = True

        target_cals = targets["target_calories"]
        actual_cals = actual["calories"]
        cal_diff_pct = ((actual_cals - target_cals) / target_cals) * 100 if target_cals > 0 else 0

        if cal_diff_pct > 15:
            feedback.append({
                "type": "warning",
                "message": f"🔥 Calorie overload: Current meal plan exceeds your daily target by {cal_diff_pct:.0f}% (+{int(actual_cals - target_cals)} kcal)."
            })
            is_compliant = False
        elif cal_diff_pct < -20:
            feedback.append({
                "type": "info",
                "message": f"🥗 Calorie deficit: You are {abs(cal_diff_pct):.0f}% below target. Ensure you eat enough to sustain energy."
            })

        # Check macro grams compliance
        macro_targets = targets["macro_targets"]
        for key, name, unit, dense in [("protein", "Protein", "g", 4.0), ("carbs", "Carbs", "g", 4.0), ("fat", "Fat", "g", 9.0)]:
            act_val = actual.get(key, 0.0)
            tar_val = macro_targets.get(key, 0.0)
            if tar_val <= 0:
                continue

            diff = act_val - tar_val
            pct_diff = (diff / tar_val) * 100

            # Warn on severe deviations (e.g. low protein or high carbs on keto)
            if key == "protein" and pct_diff < -20:
                feedback.append({
                    "type": "warning",
                    "message": f"💪 Protein shortage: You are short of your protein target by {abs(diff):.1f}g. Consider adding high-protein ingredients."
                })
                is_compliant = False
            elif key == "carbs" and pct_diff > 30 and targets["macro_percentages"]["carbs"] <= 15:
                # Carbs over on a low carb/keto diet
                feedback.append({
                    "type": "warning",
                    "message": f"⚠️ Carb Limit Breached: On your current low-carb profile, you exceed carbs by {diff:.1f}g. Look for grain/sugar alternatives."
                })
                is_compliant = False
            elif key == "fat" and pct_diff > 25:
                feedback.append({
                    "type": "info",
                    "message": f"🥑 Fat levels: Current plan is {diff:.1f}g above target fat levels. Watch oils/butters if trying to cut."
                })

        if is_compliant and feedback == []:
            feedback.append({
                "type": "success",
                "message": "🌟 Nutrition Compliance: Your current daily meal plan aligns perfectly with your macro and calorie goal profiles."
            })

        return {
            "compliant": is_compliant,
            "alerts": feedback
        }

    @classmethod
    def generate_dietitian_system_prompt(cls, user_profile: Dict[str, Any], targets: Dict[str, Any], daily_plan_summary: str) -> str:
        """
        Builds the detailed dietitian persona prompt containing user parameters and constraints.
        """
        return f"""You are a registered clinical dietitian, sports nutritionist, and health coach.
Your primary role is to advise the user on how to optimize their meals to hit their specific health goals.

USER PROFILE:
- Gender: {user_profile.get('gender', 'Not set')}
- Age: {user_profile.get('age', 'Not set')} years
- Weight: {user_profile.get('weight', 'Not set')} kg
- Height: {user_profile.get('height', 'Not set')} cm
- Activity Level: {user_profile.get('activity', 'Not set')}
- Primary Goal: {user_profile.get('goal', 'Not set')}

CALORIE & MACRO DAILY TARGETS:
- Target Calories: {targets.get('target_calories', 2000)} kcal
- Carb Target: {targets.get('macro_targets', {}).get('carbs', 0)}g ({targets.get('macro_percentages', {}).get('carbs', 0)}%)
- Protein Target: {targets.get('macro_targets', {}).get('protein', 0)}g ({targets.get('macro_percentages', {}).get('protein', 0)}%)
- Fat Target: {targets.get('macro_targets', {}).get('fat', 0)}g ({targets.get('macro_percentages', {}).get('fat', 0)}%)

CURRENT DAILY MEAL PLAN SUMMARY:
{daily_plan_summary}

RULES OF CONVERSATION:
1. Always be highly encouraging, empathetic, and scientific. Use clinical nutrition rationale.
2. Provide concrete, actionable substitutions based on their active meal plan (e.g. "Instead of using whole eggs in the breakfast omelet, swap for egg whites to reduce fat by 10g while maintaining protein").
3. Suggest ingredient adjustments that help bridge their gaps (e.g., if low on protein, suggest adding 150g grilled chicken, tofu, or protein supplement).
4. If they ask general health/nutrition questions (e.g. "is aspartame safe?" or "what are clean carbs?"), explain clearly with medical facts, structured bullet points, and concise language.
5. Keep your responses structured, clean, and concise (under 250 words per response).
"""

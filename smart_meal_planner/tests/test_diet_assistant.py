"""
Unit tests for DietAssistantModule
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.diet_assistant import DietAssistantModule


class TestDietAssistantModule:

    def test_calculate_bmr_male(self):
        # Male Mifflin-St Jeor: 10 * 80 + 6.25 * 180 - 5 * 30 + 5 = 800 + 1125 - 150 + 5 = 1780
        bmr = DietAssistantModule.calculate_bmr(
            weight_kg=80.0,
            height_cm=180.0,
            age_years=30,
            gender="male"
        )
        assert bmr == 1780.0

    def test_calculate_bmr_female(self):
        # Female Mifflin-St Jeor: 10 * 60 + 6.25 * 165 - 5 * 25 - 161 = 600 + 1031.25 - 125 - 161 = 1345.25
        bmr = DietAssistantModule.calculate_bmr(
            weight_kg=60.0,
            height_cm=165.0,
            age_years=25,
            gender="female"
        )
        assert bmr == 1345.25

    def test_calculate_tdee(self):
        # TDEE for BMR 1780 and Moderately Active (1.55) = 1780 * 1.55 = 2759.0
        tdee = DietAssistantModule.calculate_tdee(
            bmr=1780.0,
            activity_level="Moderately Active (Exercise 3-5 days/week)"
        )
        assert tdee == 2759.0

    def test_calculate_targets_weight_loss(self):
        # User: 80kg, 180cm, 30yo, Male, Moderately Active, Weight Loss, Keto
        # BMR = 1780
        # TDEE = 1780 * 1.55 = 2759
        # Target Calories (Deficit 15%) = 2759 * 0.85 = 2345
        # Keto Split: 5% carbs, 25% protein, 70% fat
        # Carb Cal = 2345 * 0.05 = 117.25 kcal -> /4 = 29.3 g
        # Protein Cal = 2345 * 0.25 = 586.25 kcal -> /4 = 146.6 g
        # Fat Cal = 2345 * 0.70 = 1641.5 kcal -> /9 = 182.4 g
        targets = DietAssistantModule.calculate_targets(
            weight_kg=80.0,
            height_cm=180.0,
            age_years=30,
            gender="male",
            activity_level="Moderately Active (Exercise 3-5 days/week)",
            diet_goal="Weight Loss (Moderate deficit)",
            macro_preset="Keto (5/25/70)"
        )
        assert targets["bmr"] == 1780.0
        assert targets["tdee"] == 2759.0
        assert targets["target_calories"] == 2345.0
        assert targets["macro_targets"]["carbs"] == 29.3
        assert targets["macro_targets"]["protein"] == 146.6
        assert targets["macro_targets"]["fat"] == 182.4

    def test_get_compliance_feedback_compliant(self):
        targets = {
            "target_calories": 2000.0,
            "macro_targets": {"carbs": 250.0, "protein": 125.0, "fat": 56.0},
            "macro_percentages": {"carbs": 50, "protein": 25, "fat": 25}
        }
        actual = {
            "calories": 1950.0,
            "carbs": 240.0,
            "protein": 120.0,
            "fat": 55.0
        }
        feedback = DietAssistantModule.get_compliance_feedback(actual, targets)
        assert feedback["compliant"] is True
        assert len(feedback["alerts"]) == 1
        assert "Nutrition Compliance" in feedback["alerts"][0]["message"]

    def test_get_compliance_feedback_non_compliant(self):
        # Low protein warning and excess calorie warning
        targets = {
            "target_calories": 1500.0,
            "macro_targets": {"carbs": 187.5, "protein": 93.8, "fat": 41.7},
            "macro_percentages": {"carbs": 50, "protein": 25, "fat": 25}
        }
        actual = {
            "calories": 2200.0,
            "carbs": 250.0,
            "protein": 50.0,
            "fat": 80.0
        }
        feedback = DietAssistantModule.get_compliance_feedback(actual, targets)
        assert feedback["compliant"] is False
        alerts = [a["message"] for a in feedback["alerts"]]
        assert any("Calorie overload" in msg for msg in alerts)
        assert any("Protein shortage" in msg for msg in alerts)

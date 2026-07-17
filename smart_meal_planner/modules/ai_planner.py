"""
AI Live Planner Module
======================
Generates fully custom day plans (breakfast, lunch, dinner) 
dynamically using LLMs, bypassing the predefined CSV dataset.
"""

import os
import json
import logging
import concurrent.futures
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)

class AILivePlanner:
    def __init__(self, api_key: str = None, provider: str = "auto"):
        self._available = False
        self._provider = None
        self._client = None

        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        if gemini_key and not gemini_key.startswith("AIzaSy"):
            gemini_key = ""

        if provider == "openai" and openai_key:
            self._init_openai(openai_key)
        elif provider == "groq" and groq_key:
            self._init_groq(groq_key)
        elif provider == "gemini" and gemini_key:
            self._init_gemini(gemini_key)
        elif provider == "auto":
            if groq_key: self._init_groq(groq_key)
            elif gemini_key: self._init_gemini(gemini_key)
            elif openai_key: self._init_openai(openai_key)

    def _init_openai(self, key):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
            self._provider = "openai"
            self._available = True
            logger.info("AILivePlanner: Initialized OpenAI.")
        except ImportError:
            logger.warning("OpenAI SDK not found.")

    def _init_groq(self, key):
        try:
            from groq import Groq
            self._client = Groq(api_key=key)
            self._provider = "groq"
            self._available = True
            logger.info("AILivePlanner: Initialized Groq.")
        except ImportError:
            logger.warning("Groq SDK not found.")

    def _init_gemini(self, key):
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
            self._provider = "gemini"
            self._available = True
            logger.info("AILivePlanner: Initialized Gemini.")
        except ImportError:
            logger.warning("Gemini SDK not found.")

    def generate_single_recipe(
        self,
        user_ingredients: list,
        meal_type: str,
        diet_goal: str = "Balanced",
        calorie_limit: int = 600,
        dietary_tags: list = None,
    ) -> dict:
        if not self._available:
            return None

        tags_str = ", ".join(dietary_tags) if dietary_tags else "none"
        prompt = f"""You are a professional nutritionist and Michelin-star chef specializing in CREATING recipes from whatever ingredients a person has at home.

Your ONLY job here is to MAKE a recipe using the ingredients the user already has. Do NOT suggest recipes that require ingredients the user doesn't have.

User's available ingredients: {", ".join(user_ingredients[:25]) if user_ingredients else "Anything"}
- Meal type: {meal_type}
- Diet / goal: {diet_goal}
- Maximum calories: {calorie_limit} kcal per serving
- Dietary requirements: {tags_str}

STRICT RULES:
1. BUILD the recipe ONLY from the user's listed ingredients above.
2. You may add AT MOST 1-2 absolute basic pantry staples (salt, black pepper, water, or cooking oil) — nothing else.
3. Do NOT add extra proteins, vegetables, spices, sauces, or ANY ingredient not in the user's list.
4. The recipe MUST make practical sense with those exact ingredients.
5. If the ingredients suggest a well-known dish (e.g. 'Vada Pav', 'Pasta'), make that authentic dish using only what's available.
6. Calories MUST be <= {calorie_limit}.

Return ONLY a valid JSON object with this EXACT structure (no markdown, no explanation):
{{
  "title": "Recipe Name",
  "cuisine": "cuisine type",
  "meal_type": "{meal_type}",
  "ingredients": "ingredient1|ingredient2|ingredient3",
  "instructions": "1. Step one action here. 2. Step two action here. 3. Step three action here. 4. Step four action here. 5. Final plating step.",
  "tags": "tag1,tag2,tag3",
  "calories": 450,
  "protein": 35.0,
  "carbs": 40.0,
  "fat": 12.0,
  "prep_time_min": 20
}}

- instructions MUST have at least 4 numbered steps ("1. ...", "2. ...", etc.), each step a clear cooking action.
- ingredients field MUST be pipe-separated lowercase names (e.g. chicken_breast|garlic|olive_oil).
- tags field MUST be comma-separated.
"""
        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                raw_text = response.choices[0].message.content.strip()
            elif self._provider == "groq":
                response = self._client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                raw_text = response.choices[0].message.content.strip()
            else:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.7},
                )
                raw_text = response.text.strip()

            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("```").strip()
            obj_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if obj_match:
                recipe_json = json.loads(obj_match.group(0))
                # The frontend and backend serializers expect `ingredient_list` as an Array
                if "ingredients" in recipe_json and isinstance(recipe_json["ingredients"], str):
                    recipe_json["ingredient_list"] = [i.strip() for i in recipe_json["ingredients"].split('|') if i.strip()]
                return recipe_json
            return None
        except Exception as e:
            logger.error(f"AILivePlanner generation error for {meal_type}: {e}")
            return None

    def plan_day(
        self,
        user_ingredients: List[str],
        dietary_filters: Optional[List[str]] = None,
        indian_focus: bool = False,
    ) -> Dict[str, Optional[Dict]]:
        
        dietary_tags = list(dietary_filters) if dietary_filters else []
        if indian_focus:
            dietary_tags.append("indian")

        plan = {}
        slots = ["breakfast", "lunch", "dinner"]
        
        # Run generations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_slot = {
                executor.submit(
                    self.generate_single_recipe,
                    user_ingredients,
                    slot,
                    "Balanced",
                    500 if slot == "breakfast" else 700,
                    dietary_tags
                ): slot for slot in slots
            }
            
            for future in concurrent.futures.as_completed(future_to_slot):
                slot = future_to_slot[future]
                try:
                    recipe = future.result()
                    # Add missing_ingredients field for the UI to not throw errors
                    if recipe:
                        recipe["missing_ingredients"] = []
                    plan[slot] = recipe
                except Exception as e:
                    logger.error(f"Exception in AILivePlanner parallel generation for {slot}: {e}")
                    plan[slot] = None

        return plan

    def generate_alternatives(
        self,
        user_ingredients: List[str],
        dietary_filters: Optional[List[str]] = None,
        indian_focus: bool = False,
        count: int = 5
    ) -> List[Dict]:
        if not self._available:
            return []

        dietary_tags = list(dietary_filters) if dietary_filters else []
        if indian_focus:
            dietary_tags.append("indian")
            
        tags_str = ", ".join(dietary_tags) if dietary_tags else "none"
        prompt = f"""You are a professional nutritionist and Michelin-star chef specializing in CREATING recipes from whatever ingredients a person has at home.

Your ONLY job is to MAKE {count} DIFFERENT recipes using ONLY the user's listed ingredients. Do NOT suggest recipes that require ingredients the user doesn't have.

User's available ingredients: {", ".join(user_ingredients[:25]) if user_ingredients else "Anything"}
- Diet / goal: Balanced
- Dietary requirements: {tags_str}

STRICT RULES:
1. EVERY recipe must use ONLY the user's listed ingredients above.
2. You may add AT MOST 1-2 absolute basic pantry staples per recipe (salt, black pepper, water, or cooking oil) — nothing else from outside the list.
3. Do NOT add extra proteins, vegetables, spices, sauces, or ANY ingredient not listed by the user.
4. Each recipe MUST make practical sense with those exact ingredients.
5. Make all {count} recipes DIFFERENT from each other (different cooking methods, presentations, or combinations).
6. If the ingredients suggest a well-known dish (e.g. 'Vada Pav'), include that authentic dish as one of the options.

Return ONLY a valid JSON array of {count} objects, with each object having this EXACT structure:
{{
  "title": "Recipe Name",
  "cuisine": "cuisine type",
  "meal_type": "any",
  "ingredients": "ingredient1|ingredient2|ingredient3",
  "instructions": "1. Step one action here. 2. Step two action here. 3. Step three action here. 4. Step four action here. 5. Final plating step.",
  "tags": "tag1,tag2",
  "calories": 450,
  "protein": 35.0,
  "carbs": 40.0,
  "fat": 12.0,
  "prep_time_min": 20
}}

- Output a JSON Array: [ {{...}}, {{...}} ]
- instructions MUST have at least 4 numbered steps, each step a clear cooking action.
- ingredients field MUST be pipe-separated lowercase names.
"""
        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                raw_text = response.choices[0].message.content.strip()
            elif self._provider == "groq":
                response = self._client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                raw_text = response.choices[0].message.content.strip()
            else:
                response = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.8},
                )
                raw_text = response.text.strip()

            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("```").strip()
            obj_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if obj_match:
                recipes_array = json.loads(obj_match.group(0))
                for recipe_json in recipes_array:
                    if "ingredients" in recipe_json and isinstance(recipe_json["ingredients"], str):
                        recipe_json["ingredient_list"] = [i.strip() for i in recipe_json["ingredients"].split('|') if i.strip()]
                    recipe_json["missing_ingredients"] = []
                return recipes_array
            return []
        except Exception as e:
            logger.error(f"AILivePlanner alternatives generation error: {e}")
            return []

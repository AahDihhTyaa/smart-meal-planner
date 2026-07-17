"""
Chatbot Intent Parsing Module
=============================
Parses raw text (spoken or typed) to determine user intent, extracted ingredients,
and preferences (like 'random meal', 'dessert', etc.). Uses Groq or Gemini.
"""

import json
import logging
import os
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are the intelligence behind a Smart Meal Planner and an expert AI Chef.
The user is speaking or typing a request. You need to parse their intent.

Extract the information into a JSON object matching this schema exactly:
{
  "intent": "ADD_INGREDIENTS" | "GENERATE_MEAL" | "GENERATE_CUSTOM_RECIPE" | "CHAT",
  "ingredients": ["list", "of", "ingredients", "mentioned"],
  "meal_type": "breakfast" | "lunch" | "dinner" | "snack" | "dessert" | "any",
  "is_random": true | false,
  "chat_response": "Your conversational response if the intent is CHAT. Keep it helpful, culinary-focused, and concise."
}

RULES:
- If the user just lists ingredients (e.g., "apple, banana, milk") or says "I have...", intent is "ADD_INGREDIENTS".
- If the user asks for a meal, recipe, or plan from existing data, intent is "GENERATE_MEAL".
- If the user explicitly asks to "invent", "create a custom recipe", or "generate a new recipe" from scratch using their ingredients, intent is "GENERATE_CUSTOM_RECIPE".
- If the user asks a general cooking question, substitution advice, or just chats, intent is "CHAT" and provide a helpful `chat_response`.
- Normalize ingredient names (e.g., lowercased, singular).
- If the user explicitly asks for "random", set is_random to true.
- If they specify a time of day (breakfast/lunch/dinner) or dessert, set meal_type.
- Output ONLY the raw JSON string. Do not include markdown blocks like ```json ... ```. Do not include any explanations.
"""

class ChatbotIntentParser:
    def __init__(self, api_key: str = None, provider: str = "auto"):
        self._available = False
        self._provider = None
        self._client = None

        groq_key = os.getenv("GROQ_API_KEY", "")
        gemini_key = api_key or os.getenv("GEMINI_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")

        # Sanitize Gemini API Key
        if gemini_key and not gemini_key.startswith("AIzaSy"):
            logger.warning("Ignoring invalid Gemini API key format (must start with AIzaSy).")
            gemini_key = ""

        # Force provider logic
        if provider == "openai" and openai_key:
            self._init_openai(openai_key)
        elif provider == "groq" and groq_key:
            self._init_groq(groq_key)
        elif provider == "gemini" and gemini_key:
            self._init_gemini(gemini_key)
        else:
            # Auto fallback logic
            if groq_key:
                self._init_groq(groq_key)
            if not self._available and openai_key:
                self._init_openai(openai_key)
            if not self._available and gemini_key:
                self._init_gemini(gemini_key)

    def _init_openai(self, key: str):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
            self._provider = "openai"
            self._available = True
        except ImportError:
            pass
        except Exception as e:
            logger.warning("OpenAI init failed: %s", e)

    def _init_groq(self, key: str):
        try:
            from groq import Groq
            self._client = Groq(api_key=key)
            self._provider = "groq"
            self._available = True
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    def _init_gemini(self, key: str):
        try:
            from google import genai
            self._client = genai.Client(api_key=key)
            self._provider = "gemini"
            self._available = True
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    def parse_intent(self, user_text: str) -> Dict[str, Any]:
        """
        Calls the LLM to parse intent. Returns a dictionary.
        Returns a default ADD_INGREDIENTS dict if it fails.
        """
        default_resp = {
            "intent": "ADD_INGREDIENTS",
            "ingredients": [],
            "meal_type": "any",
            "is_random": False
        }

        if not self._available:
            return default_resp

        try:
            if self._provider == "openai":
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.0
                )
                raw_text = response.choices[0].message.content
            elif self._provider == "groq":
                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.0
                )
                raw_text = response.choices[0].message.content
            else:
                response = self._client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=f"{_SYSTEM_PROMPT}\n\nUser text: {user_text}",
                    config={"temperature": 0.0}
                )
                raw_text = response.text

            # Clean up markdown if the LLM accidentally added it
            raw_text = raw_text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.MULTILINE).strip()

            parsed = json.loads(raw_text)
            
            # Merge with default to ensure keys exist
            return {**default_resp, **parsed}
            
        except Exception as e:
            logger.error(f"ChatbotIntentParser error: {e}")
            return default_resp

    def chat_with_chef(self, messages: list, system_msg: str = None) -> str:
        """
        Sends the full chat history to the LLM to get a smart, context-aware response.
        """
        if not self._available:
            return "I'm sorry, my AI brain is currently disconnected. Please check your API keys."
            
        if system_msg is None:
            system_msg = "You are an expert, Michelin-star AI Chef assisting a user with their Smart Meal Planner. Be helpful, incredibly smart about culinary science, substitutions, and recipes. Keep responses concise, highly professional, and insightful."
        
        try:
            if self._provider == "openai":
                openai_messages = [{"role": "system", "content": system_msg}]
                for m in messages:
                    if m["role"] in ["user", "assistant"]:
                        openai_messages.append({"role": m["role"], "content": m["content"]})
                
                response = self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=openai_messages,
                    temperature=0.6
                )
                return response.choices[0].message.content
            elif self._provider == "groq":
                groq_messages = [{"role": "system", "content": system_msg}]
                for m in messages:
                    # Ignore tool or non-standard roles for safety
                    if m["role"] in ["user", "assistant"]:
                        groq_messages.append({"role": m["role"], "content": m["content"]})
                    
                response = self._client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=groq_messages,
                    temperature=0.6
                )
                return response.choices[0].message.content
            else:
                # Gemini
                from google.genai import types
                gemini_messages = []
                for m in messages:
                    role = "user" if m["role"] == "user" else "model"
                    gemini_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))
                    
                response = self._client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=gemini_messages,
                    config=types.GenerateContentConfig(
                        system_instruction=system_msg,
                        temperature=0.6
                    )
                )
                return response.text
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return "I ran into a small issue processing that request in my kitchen. Could you ask again?"

    def build_context_aware_system_prompt(self, user_ingredients: list, active_plan: dict, diet_profile: dict, seasonal_info: str, active_recipe: dict = None, current_step: int = 0) -> str:
        """
        Builds a rich system prompt injecting current application state.
        """
        ingredients_str = ", ".join(user_ingredients) if user_ingredients else "None currently in pantry."
        
        plan_str = "No active meal plan today."
        if active_plan:
            plan_str = "\n".join([f"- {slot.capitalize()}: {recipe['title'] if recipe else 'Unplanned'}" 
                                  for slot, recipe in active_plan.items()])
                                  
        goal = diet_profile.get("goal", "Balanced")
        calorie_target = diet_profile.get("target_calories", 2000)
        
        active_recipe_str = "None"
        if active_recipe:
            title = active_recipe.get("title", "Unknown")
            ingreds = active_recipe.get("ingredients", "").replace("|", ", ")
            instructions = active_recipe.get("instructions", "")
            active_recipe_str = f"Title: {title}\nIngredients: {ingreds}\nInstructions: {instructions}\nUser is currently on Step: {current_step}"
            
        return f"""You are an expert, Michelin-star AI Chef assisting a user with their Smart Meal Planner.
        
CURRENT CONTEXT:
- Pantry Ingredients: {ingredients_str}
- Today's Meal Plan:
{plan_str}
- Dietary Goal: {goal}
- Daily Calorie Target: {calorie_target} kcal
- Seasonal Context: {seasonal_info}
- Active Recipe (Currently Cooking):
{active_recipe_str}

RULES FOR YOUR RESPONSES:
1. You MUST factor in the user's current pantry ingredients when suggesting what to cook.
2. If they ask "what can I make?", suggest meals using what they have, or mention missing staples.
3. If they ask about nutrition, reference their dietary goal ({goal}) and calorie target.
4. Keep responses conversational, highly professional, culinary-focused, and concise (under 200 words).
5. Suggest seasonal swaps or batch-cooking opportunities if relevant.
6. If an active recipe is provided, use its ingredients, instructions, and current step to answer questions like "How much do I need?" or "What's the next step?".
"""

    def generate_smart_suggestions(self, user_ingredients: list, active_plan: dict) -> list:
        """
        Proactively generate 3 short, actionable questions the user can ask the AI.
        """
        suggestions = ["🥗 What can I make right now?"]
        
        if user_ingredients:
            suggestions.append("🔄 What staple ingredients am I missing?")
            
        unplanned = [slot for slot, recipe in active_plan.items() if recipe is None] if active_plan else ["breakfast", "lunch", "dinner"]
        if unplanned:
            suggestions.append(f"💡 Suggest a custom {unplanned[0]} recipe")
        else:
            suggestions.append("💪 How well does my plan hit my macros?")
            
        return suggestions[:3]

    def generate_custom_recipe(
        self,
        user_ingredients: list,
        meal_type: str = "any",
        diet_goal: str = "Balanced",
        calorie_limit: int = 600,
        dietary_tags: list = None,
    ) -> dict:
        """
        Ask the LLM to generate a completely personalized recipe from scratch,
        tailored to the user's available ingredients, calorie limits, and diet goal.

        Returns a dict matching the recipe corpus structure:
            title, ingredients (pipe-separated), instructions, meal_type,
            calories, protein, carbs, fat, tags, cuisine
        """
        if not self._available:
            return {}

        tags_str = ", ".join(dietary_tags) if dietary_tags else "none"
        prompt = f"""You are a professional nutritionist and Michelin-star chef.

Generate ONE complete recipe using these constraints:
- Available ingredients: {", ".join(user_ingredients[:25])}
- Meal type: {meal_type}
- Diet / goal: {diet_goal}
- Maximum calories: {calorie_limit} kcal per serving
- Dietary requirements: {tags_str}

Return ONLY a valid JSON object with this EXACT structure (no markdown, no explanation):
{{
  "title": "Recipe Name",
  "cuisine": "cuisine type",
  "meal_type": "{meal_type if meal_type != 'any' else 'dinner'}",
  "ingredients": "ingredient1|ingredient2|ingredient3",
  "instructions": "Brief cooking instructions in 2-3 sentences.",
  "tags": "tag1,tag2,tag3",
  "calories": 450,
  "protein": 35.0,
  "carbs": 40.0,
  "fat": 12.0
}}

IMPORTANT:
- Only use ingredients from the provided list (plus basic pantry staples like salt, oil, garlic, onion, water).
- Calories MUST be ≤ {calorie_limit}. Calculate accurately based on ingredients.
- Make it creative, delicious, and nutritionally balanced for the goal: {diet_goal}.
- ingredients field: pipe-separated lowercase underscore names (e.g. chicken_breast|garlic|olive_oil).
- tags field: comma-separated (e.g. high-protein,low-carb,gluten-free).
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
                    model="llama-3.3-70b-versatile",
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

            # Clean markdown if present
            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("```").strip()
            # Extract JSON object
            obj_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if obj_match:
                return json.loads(obj_match.group(0))
            return {}
        except Exception as e:
            logger.error(f"generate_custom_recipe error: {e}")
            return {}

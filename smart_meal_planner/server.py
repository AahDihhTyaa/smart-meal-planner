import os
import sys
import base64
import json
import hashlib
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv

# Load all required modules
from modules.nlp_normalize import NLPNormalizer
from modules.recipe_matcher import RecipeMatcher
from modules.meal_planner import MealPlanner, MEAL_SLOTS, DAYS
from modules.nutrition import NutritionModule
from modules.vision_input import VisionInputModule
from modules.voice_assistant import transcribe_audio
from modules.chatbot_intent import ChatbotIntentParser
from modules.diet_assistant import DietAssistantModule
from modules.budget_optimizer import BudgetOptimizer
from modules.recommendation_engine import RecommendationEngine
from modules.cooking_coach import CookingCoach
from modules.macro_scaler import MacroScaler
from modules.analytics import PersonalAnalytics
from modules.ai_planner import AILivePlanner
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-smart-meal-planner-secret-key")

# Initialize heavy singletons
normalizer = NLPNormalizer()
matcher = RecipeMatcher()
planner = MealPlanner(matcher)
nutrition_module = NutritionModule()
diet_assistant = DietAssistantModule()
budget_optimizer = BudgetOptimizer(daily_budget=15.0)
recommendation_engine = RecommendationEngine()

def get_keys():
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key and not gemini_key.startswith("AIzaSy"):
        gemini_key = None
    return {
        'gemini': gemini_key,
        'groq': os.getenv('GROQ_API_KEY')
    }

# Lightweight Server-Side Caching
plan_cache = {}
vision_cache = {}

@app.route('/')
def index():
    return render_template('index.html')



def serialize_recipe(recipe, user_ingredients=None):
    import pandas as pd
    if recipe is None:
        return None
    
    d = None
    if isinstance(recipe, pd.Series):
        d = recipe.to_dict()
    elif isinstance(recipe, dict):
        d = dict(recipe)
        
    if d is not None:
        for k, v in d.items():
            if isinstance(v, set):
                d[k] = list(v)
            elif isinstance(v, float) and pd.isna(v):
                d[k] = None
                
        # Enrich the recipe
        if user_ingredients is not None:
            enriched = recommendation_engine.enrich_recipe(recipe, user_ingredients)
            d.update(enriched)
            
            # Calculate dynamic coverage for AI recipes
            if "ingredient_list" in d and d["ingredient_list"] and user_ingredients:
                matched_count = sum(1 for ui in user_ingredients if any(ui.lower() in str(ri).lower() for ri in d["ingredient_list"]))
                d["coverage"] = matched_count / len(user_ingredients)
            elif "coverage" not in d:
                d["coverage"] = 0.0
            
        # Add substitutions for missing ingredients
        if "missing_ingredients" in d and d["missing_ingredients"]:
            d["substitutions"] = recommendation_engine.get_substitutions(d["missing_ingredients"])
            
        return d
    return None

def serialize_day_plan(day_plan, user_ingredients=None):
    return {slot: serialize_recipe(recipe, user_ingredients) for slot, recipe in day_plan.items()}

def serialize_week_plan(week_plan, user_ingredients=None):
    return {day: serialize_day_plan(day_plan, user_ingredients) for day, day_plan in week_plan.items()}

@app.route('/api/plan', methods=['POST'])
def generate_plan():
    try:
        data = request.json
        cache_key = json.dumps(data, sort_keys=True)
        if cache_key in plan_cache:
            return jsonify(plan_cache[cache_key])
            
        ingredients_text = data.get('ingredients', '')
        
        extracted_ingredients = list(normalizer.get_ingredient_set(ingredients_text))
        if not extracted_ingredients and ingredients_text:
            extracted_ingredients = [i.strip() for i in ingredients_text.split(',') if i.strip()]
        
        dietary_filters = data.get('dietary_filters', {})
        indian_focus = data.get('indian_focus', False)
        
        keys = get_keys()
        ai_live_planner = AILivePlanner(api_key=keys['gemini'] or None, provider="auto")
        
        import concurrent.futures
        day_plan = {}
        alternatives_raw = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_day = executor.submit(ai_live_planner.plan_day, extracted_ingredients, dietary_filters, indian_focus)
            future_alt = executor.submit(ai_live_planner.generate_alternatives, extracted_ingredients, dietary_filters, indian_focus, 7)
            
            day_plan = future_day.result()
            alternatives_raw = future_alt.result()
            
        import random
        all_recipes = []
        if day_plan:
            all_recipes.extend([r for r in day_plan.values() if r])
        if alternatives_raw:
            all_recipes.extend(alternatives_raw)
            
        week_plan = {}
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        if all_recipes:
            for day in days_of_week:
                week_plan[day] = {
                    "breakfast": random.choice(all_recipes),
                    "lunch": random.choice(all_recipes),
                    "dinner": random.choice(all_recipes)
                }
        else:
            week_plan = {day: day_plan for day in days_of_week}
        
        
        matching = []
        unlockable = []
        if alternatives_raw:
            for r in alternatives_raw:
                matching.append(serialize_recipe(r, extracted_ingredients))
        
        resp = {
            'success': True,
            'day_plan': serialize_day_plan(day_plan, extracted_ingredients),
            'week_plan': serialize_week_plan(week_plan, extracted_ingredients),
            'alternatives': matching,
            'unlockable': unlockable,
            'used_ingredients': extracted_ingredients
        }
        
        plan_cache[cache_key] = resp
        if len(plan_cache) > 128:
            plan_cache.pop(next(iter(plan_cache)))
            
        return jsonify(resp)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        context = data.get('context', [])
        
        keys = get_keys()
        # Initialize parser per request to use provided keys
        intent_parser = ChatbotIntentParser(api_key=keys['gemini'] or None, provider="auto")
        
        # Read from session
        active_recipe = session.get('active_recipe', None)
        current_step = session.get('current_step', 0)
        
        intent_res = intent_parser.parse_intent(message)
        intent = intent_res.get('intent', 'CHAT')
        
        # Convert text context history (e.g. "user: text") to structured list of dicts
        messages_for_llm = []
        for item in context:
            if ":" in item:
                role, content = item.split(":", 1)
                role = "user" if role.strip().lower() == "user" else "assistant"
                messages_for_llm.append({"role": role, "content": content.strip()})
        
        messages_for_llm.append({"role": "user", "content": message})
        
        if intent == "GENERATE_CUSTOM_RECIPE":
            diet_profile = data.get('diet_profile', {})
            custom_recipe = intent_parser.generate_custom_recipe(
                user_ingredients=data.get('ingredients', []),
                meal_type="any",
                diet_goal=diet_profile.get('goal', 'Balanced'),
                calorie_limit=diet_profile.get('target_calories', 600) // 3 if diet_profile.get('target_calories') else 600
            )
            
            if custom_recipe:
                title = custom_recipe.get('title', 'Custom Recipe')
                ings = custom_recipe.get('ingredients', '').replace('|', '<br>• ')
                inst = custom_recipe.get('instructions', '')
                cals = custom_recipe.get('calories', 0)
                pro = custom_recipe.get('protein', 0)
                carbs = custom_recipe.get('carbs', 0)
                fat = custom_recipe.get('fat', 0)
                
                response_text = f"""
                <div class="mb-2"><strong>✨ {title} (AI Generated)</strong></div>
                <div class="text-sm mb-2"><strong>Ingredients:</strong><br>• {ings}</div>
                <div class="text-sm mb-2"><strong>Instructions:</strong><br>{inst}</div>
                <div class="text-xs opacity-75 mt-2 flex gap-3">
                    <span>🔥 {cals} kcal</span>
                    <span>🥩 {pro}g protein</span>
                    <span>🍞 {carbs}g carbs</span>
                    <span>🥑 {fat}g fat</span>
                </div>
                """
            else:
                response_text = "I'm sorry, I couldn't generate a custom recipe at this time."
        else:
            system_msg = intent_parser.build_context_aware_system_prompt(
                user_ingredients=data.get('ingredients', []),
                active_plan=data.get('day_plan', {}),
                diet_profile=data.get('diet_profile', {}),
                seasonal_info="Current Season",
                active_recipe=active_recipe,
                current_step=current_step
            )
            
            response_text = intent_parser.chat_with_chef(messages_for_llm, system_msg=system_msg)
        
        return jsonify({
            'success': True,
            'response': response_text,
            'intent': intent
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        day_plan = data.get('day_plan', {})
        week_plan = data.get('week_plan', {})
        
        analysis = recommendation_engine.get_full_analysis(
            user_ingredients=ingredients,
            recipe_df=matcher.corpus,
            day_plan=day_plan,
            week_plan=week_plan
        )
        return jsonify({'success': True, 'analysis': analysis})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/nutrition', methods=['POST'])
def nutrition():
    try:
        data = request.json
        title = data.get('title', '')
        ingredients = data.get('ingredients', '')
        
        nut_info = nutrition_module.get_nutrition(f"{title} {ingredients}")
        return jsonify({'success': True, 'nutrition': nut_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/vision', methods=['POST'])
def vision():
    try:
        data = request.json
        image_base64 = data.get('image_base64', '')
        mime_type = data.get('mime_type', 'image/jpeg')
        if not image_base64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        img_hash = hashlib.md5(image_base64.encode('utf-8')).hexdigest()
        if img_hash in vision_cache:
            return jsonify(vision_cache[img_hash])
        
        keys = get_keys()
        
        image_bytes = base64.b64decode(image_base64)
        
        gemini_key = keys.get('gemini')
        groq_key = keys.get('groq')
        
        # Instantiate Vision Module. Let it auto-detect via its constructor.
        vision_module = VisionInputModule(api_key=gemini_key or None, provider="auto")
        
        ingredients, err = vision_module.detect_ingredients(image_bytes, mime_type=mime_type)
        
        if err:
            return jsonify({'success': False, 'error': err}), 500
        
        resp = {'success': True, 'ingredients': ingredients}
        vision_cache[img_hash] = resp
        if len(vision_cache) > 128:
            vision_cache.pop(next(iter(vision_cache)))
            
        return jsonify(resp)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/cooking-coach', methods=['POST'])
def cooking_coach_api():
    try:
        data = request.json
        user_input = data.get('command', '')
        recipe = data.get('recipe', {})
        current_step = data.get('current_step', 0)
        
        keys = get_keys()
        coach = CookingCoach(api_key=keys['gemini'] or None, provider="auto")
        
        # Save to session
        if recipe:
            session['active_recipe'] = recipe
        session['current_step'] = current_step
        session.modified = True
        
        result = coach.handle_command(user_input, recipe, current_step)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scale-macros', methods=['POST'])
def scale_macros_api():
    try:
        data = request.json
        day_plan = data.get('day_plan', {})
        activity_level = data.get('activity_level', 'sedentary')
        
        scaled_plan = MacroScaler.scale_day_plan(day_plan, activity_level)
        return jsonify({'success': True, 'scaled_plan': scaled_plan})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics', methods=['POST'])
def analytics_api():
    try:
        data = request.json
        week_plan = data.get('week_plan', {})
        
        analytics_engine = PersonalAnalytics(budget_optimizer)
        report = analytics_engine.generate_full_report(week_plan)
        
        if "error" in report:
            return jsonify({'success': False, 'error': report["error"]}), 400
            
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5001")

    Timer(1, open_browser).start()
    app.run(debug=True, port=5001, use_reloader=False)

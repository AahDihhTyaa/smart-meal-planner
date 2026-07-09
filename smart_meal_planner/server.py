import os
import sys
import base64
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template, request, jsonify
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

load_dotenv()

app = Flask(__name__)

# Initialize heavy singletons
normalizer = NLPNormalizer()
matcher = RecipeMatcher()
planner = MealPlanner(matcher)
nutrition_module = NutritionModule()
diet_assistant = DietAssistantModule()
budget_optimizer = BudgetOptimizer(daily_budget=15.0)
recommendation_engine = RecommendationEngine()

def get_keys():
    return {
        'gemini': request.headers.get('X-Gemini-Key', ''),
        'groq': request.headers.get('X-Groq-Key', '')
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/viz')
def viz():
    try:
        root_dir = Path(__file__).parent
        viz_path = root_dir / 'architecture_viz.html'
        if viz_path.exists():
            with open(viz_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "Visualizer file not found", 404
    except Exception as e:
        return str(e), 500



def serialize_recipe(recipe):
    import pandas as pd
    if recipe is None:
        return None
    if isinstance(recipe, pd.Series):
        d = recipe.to_dict()
        for k, v in d.items():
            if isinstance(v, set):
                d[k] = list(v)
            elif isinstance(v, float) and pd.isna(v):
                d[k] = None
        return d
    if isinstance(recipe, dict):
        return recipe
    return None

def serialize_day_plan(day_plan):
    return {slot: serialize_recipe(recipe) for slot, recipe in day_plan.items()}

def serialize_week_plan(week_plan):
    return {day: serialize_day_plan(day_plan) for day, day_plan in week_plan.items()}

@app.route('/api/plan', methods=['POST'])
def generate_plan():
    try:
        data = request.json
        ingredients_text = data.get('ingredients', '')
        
        extracted_ingredients = list(normalizer.get_ingredient_set(ingredients_text))
        if not extracted_ingredients and ingredients_text:
            extracted_ingredients = [i.strip() for i in ingredients_text.split(',') if i.strip()]
        
        dietary_filters = data.get('dietary_filters', {})
        indian_focus = data.get('indian_focus', False)
        
        day_plan = planner.plan_day(extracted_ingredients, dietary_filters=dietary_filters, indian_focus=indian_focus)
        week_plan = planner.plan_week(extracted_ingredients, dietary_filters=dietary_filters, indian_focus=indian_focus)
        
        # Get candidates for alternatives and unlockables
        used_today = {recipe['title'] for recipe in day_plan.values() if recipe is not None}
        all_candidates = matcher.match(
            user_ingredients=extracted_ingredients,
            meal_type="any",
            dietary_filters=dietary_filters,
            top_n=30,
            indian_focus=indian_focus
        )
        
        matching = []
        unlockable = []
        if not all_candidates.empty:
            for _, r in all_candidates.iterrows():
                if r['title'] in used_today: continue
                
                if r['missing_count'] == 0 and len(matching) < 5:
                    matching.append(serialize_recipe(r))
                elif 0 < r['missing_count'] <= 3 and len(unlockable) < 5:
                    unlockable.append(serialize_recipe(r))
        
        return jsonify({
            'success': True,
            'day_plan': serialize_day_plan(day_plan),
            'week_plan': serialize_week_plan(week_plan),
            'alternatives': matching,
            'unlockable': unlockable,
            'used_ingredients': extracted_ingredients
        })

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
        
        response_text = intent_parser.chat_with_chef(messages_for_llm)
        
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
        
        keys = get_keys()
        
        image_bytes = base64.b64decode(image_base64)
        
        gemini_key = keys.get('gemini')
        groq_key = keys.get('groq')
        
        # Instantiate Vision Module with provided keys
        vision_module = VisionInputModule(api_key=gemini_key or groq_key or None, provider="auto")
        
        if groq_key:
            vision_module._init_groq(groq_key)
        elif gemini_key:
            vision_module._init_gemini(gemini_key)
            
        ingredients, err = vision_module.detect_ingredients(image_bytes, mime_type=mime_type)
        
        if err:
            return jsonify({'success': False, 'error': err}), 500
        
        return jsonify({'success': True, 'ingredients': ingredients})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    from threading import Timer

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5001")

    Timer(1, open_browser).start()
    app.run(debug=True, port=5001, use_reloader=False)

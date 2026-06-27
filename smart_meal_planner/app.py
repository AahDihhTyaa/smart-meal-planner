"""
Smart Meal Planner — Streamlit Application
==========================================
Run with:  streamlit run app.py
"""

import os
import logging
# Triggering hot-reload to unleash the ultimate Aurelia Ultra CSS
from collections import defaultdict
import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

from modules.nlp_normalize import NLPNormalizer
from modules.recipe_matcher import RecipeMatcher
from modules.meal_planner import MealPlanner, MEAL_SLOTS, DAYS
from modules.nutrition import NutritionModule
from modules.vision_input import VisionInputModule
from modules.voice_assistant import LANGUAGES, get_voice_widget_html, transcribe_audio, speak_text_html, get_read_aloud_button_html
from modules.chatbot_intent import ChatbotIntentParser
from modules.diet_assistant import DietAssistantModule
from modules.budget_optimizer import BudgetOptimizer
from modules.recommendation_engine import RecommendationEngine
import streamlit.components.v1 as components

import contextlib

@contextlib.contextmanager
def custom_spinner(text, keep_alive=False):
    placeholder = st.empty()
    placeholder.markdown(f'''
        <div class="fullscreen-overlay"></div>
        <div class="skeleton-loader-container fullscreen-centered" style="align-items: center; justify-content: center; text-align: center; padding: 3rem;">
            <div class="skeleton-icon-wrapper" style="flex-direction: column; gap: 1.5rem;">
                <div class="skeleton-icon" style="font-size: 4rem;">🍱</div>
                <div class="skeleton-text" style="font-size: 1.5rem;">{text}</div>
                <div style="font-size: 0.9rem; color: var(--text-muted); margin-top: 1rem;">Please wait while the AI analyzes your ingredients...</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)
    try:
        yield
    finally:
        if not keep_alive:
            placeholder.empty()


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Meal Planner",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")


@st.cache_data
def load_css() -> str:
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


def inject_ui_enhancements():
    """Smooth scroll, scroll-reveal, progress bars — injected once per session."""
    if st.session_state.get("_ui_enhancements_loaded"):
        return
    st.session_state._ui_enhancements_loaded = True
    components.html(
        """
<script>
(function () {
  const doc = window.parent.document;
  if (doc.getElementById("smp-ui-enhancements")) return;

  const marker = doc.createElement("div");
  marker.id = "smp-ui-enhancements";
  doc.body.appendChild(marker);

  const reducedMotion = window.parent.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let refreshPending = false;

  function scrollContainer() {
    return (
      doc.querySelector('[data-testid="stAppViewContainer"]') ||
      doc.querySelector("section.main") ||
      doc.querySelector(".main") ||
      doc.documentElement
    );
  }

  function smoothScrollTo(top) {
    const el = scrollContainer();
    if (!el) return;
    el.scrollTo({ top, behavior: reducedMotion ? "auto" : "smooth" });
  }

  function animateCoverageBars() {
    doc.querySelectorAll(".coverage-bar-inner, .progress-bar-fill").forEach(function (bar) {
      const target = bar.getAttribute("data-fill") || bar.style.width || "0%";
      if (bar.dataset.animated === "1") return;
      bar.dataset.animated = "1";
      bar.style.width = "0%";
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          bar.style.width = target;
        });
      });
    });
  }

  function animateConfidenceRings() {
    doc.querySelectorAll(".confidence-ring .ring-fill").forEach(function (ring) {
      if (ring.dataset.animated === "1") return;
      const target = ring.getAttribute("stroke-dashoffset");
      const dasharray = ring.getAttribute("stroke-dasharray");
      if (!target || !dasharray) return;
      ring.dataset.animated = "1";
      ring.style.strokeDashoffset = dasharray;
      requestAnimationFrame(function () {
        ring.style.strokeDashoffset = target;
      });
    });
  }

  const revealObserver = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -30px 0px" }
  );

  const revealSelectors =
    ".hero-banner, .section-header, .recipe-card, .confidence-ring-container, " +
    ".week-cell, .kitchen-tips-card, .empty-state, .chip-container, .day-header, .unlock-card, .tip-item";

  function bindScrollReveal() {
    doc.querySelectorAll(revealSelectors).forEach(function (el, index) {
      if (el.classList.contains("scroll-reveal")) return;
      el.classList.add("scroll-reveal");
      el.style.setProperty("--reveal-delay", Math.min(index * 0.04, 0.35) + "s");
      if (!reducedMotion) revealObserver.observe(el);
      else el.classList.add("is-visible");
    });

    doc.querySelectorAll(".stat-card").forEach(function (el, index) {
      if (el.classList.contains("scroll-reveal-scale")) return;
      el.classList.add("scroll-reveal-scale");
      el.style.setProperty("--reveal-delay", Math.min(index * 0.06, 0.4) + "s");
      if (!reducedMotion) revealObserver.observe(el);
      else el.classList.add("is-visible");
    });
  }

  function bindTabSmoothScroll() {
    doc.querySelectorAll('[data-baseweb="tab"]').forEach(function (tab) {
      if (tab.dataset.smpBound) return;
      tab.dataset.smpBound = "1";
      tab.addEventListener("click", function () {
        setTimeout(function () { smoothScrollTo(0); }, 80);
      });
    });
  }

  function scrollChatToBottom() {
    const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
    if (messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.dataset.smpScrolled) return;
    last.dataset.smpScrolled = "1";
    setTimeout(function () {
      last.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "end" });
    }, 120);
  }

  function refresh() {
    bindScrollReveal();
    bindTabSmoothScroll();
    animateCoverageBars();
    animateConfidenceRings();
    scrollChatToBottom();
  }

  let refreshTimer = null;
  function scheduleRefresh() {
    if (refreshTimer) return;
    refreshTimer = setTimeout(function () {
      refreshTimer = null;
      requestAnimationFrame(refresh);
    }, 150);
  }

  const domObserver = new MutationObserver(scheduleRefresh);
  domObserver.observe(doc.body, { childList: true, subtree: true });
  refresh();
})();
</script>
        """,
        height=0,
    )










# ── Cached resource initialization ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading NLP models…")
def load_normalizer():
    return NLPNormalizer()


@st.cache_resource(show_spinner="Loading recipe database…")
def load_matcher():
    return RecipeMatcher()


@st.cache_resource
def load_planner(_matcher):
    return MealPlanner(_matcher)


@st.cache_resource
def load_nutrition():
    return NutritionModule()


@st.cache_resource
def load_diet_assistant():
    return DietAssistantModule()


@st.cache_resource
def load_budget_optimizer(daily_budget: float = 15.0):
    return BudgetOptimizer(daily_budget=daily_budget)


@st.cache_resource
def load_recommendation_engine():
    return RecommendationEngine()


@st.cache_resource
def load_vision(gemini_key: str = "", groq_key: str = "", openai_key: str = "", provider: str = "auto"):
    """Cached — busts when API keys or provider change."""
    return VisionInputModule(api_key=gemini_key or None, provider=provider)


@st.cache_resource
def load_intent_parser(gemini_key: str = "", groq_key: str = "", openai_key: str = "", provider: str = "auto"):
    """Cached — busts when API keys or provider change."""
    return ChatbotIntentParser(api_key=gemini_key or None, provider=provider)


def get_effective_day_plan():
    """Merge base day plan with per-slot regenerations."""
    day_plan = dict(st.session_state.day_plan or {})
    for slot, recipe in (st.session_state.day_plan_overrides or {}).items():
        day_plan[slot] = recipe
    return day_plan


def _plan_titles_fingerprint(day_plan, week_plan):
    def day_fp(plan):
        if not plan:
            return ()
        return tuple(
            sorted(
                (slot, (recipe.get("title") if recipe is not None else None))
                for slot, recipe in plan.items()
            )
        )

    week_fp = ()
    if week_plan:
        week_fp = tuple(
            sorted(
                (day, slot, (recipe.get("title") if recipe is not None else None))
                for day, dp in week_plan.items()
                if isinstance(dp, dict)
                for slot, recipe in dp.items()
            )
        )
    return day_fp(day_plan), week_fp


def invalidate_analysis_cache():
    st.session_state.pop("_analysis_cache_key", None)
    st.session_state.pop("_analysis_cache", None)


def invalidate_enrichment_cache():
    st.session_state.pop("_enrichment_cache", None)


def get_cached_analysis(ingredients, day_plan, week_plan, rec_engine, matcher):
    cache_key = (
        tuple(sorted(ingredients)),
        _plan_titles_fingerprint(day_plan, week_plan),
    )
    if st.session_state.get("_analysis_cache_key") == cache_key:
        return st.session_state["_analysis_cache"]
    analysis = rec_engine.get_full_analysis(
        user_ingredients=ingredients,
        recipe_df=matcher.corpus,
        day_plan=day_plan,
        week_plan=week_plan,
    )
    st.session_state["_analysis_cache_key"] = cache_key
    st.session_state["_analysis_cache"] = analysis
    return analysis


def get_recipe_enrichment(recipe, user_ingredients, rec_engine):
    title = recipe.get("title", "")
    cache = st.session_state.setdefault("_enrichment_cache", {})
    if title in cache:
        return cache[title]
    enrichment = rec_engine.enrich_recipe(recipe, user_ingredients)
    cache[title] = enrichment
    return enrichment


def _quick_chat_intent(text: str):
    """Fast local routing — avoids an extra LLM call for plain chat."""
    lower = text.lower()
    if any(k in lower for k in ("generate", "meal plan", "make me a", "plan my", "what can i cook")):
        return "GENERATE_MEAL"
    if any(k in lower for k in ("i have", "add ", "my pantry", "ingredients:")):
        return "ACTION"
    if "," in text and not text.strip().endswith("?"):
        return "ACTION"
    return "CHAT"


def _extract_ingredients_from_text(raw_text, parsed, normalizer):
    """LLM first, NLP fallback when offline or parse returned nothing."""
    extracted = parsed.get("ingredients", [])
    if extracted:
        return extracted
    if not raw_text.strip():
        return []
    nlp_set = normalizer.get_ingredient_set(raw_text)
    return list(nlp_set) if nlp_set else []


def _apply_meal_plan(ings, planner, dietary_filters, indian_focus):
    """Generate and store day/week plans; invalidate derived caches."""
    day = planner.plan_day(ings, dietary_filters=dietary_filters, indian_focus=indian_focus)
    week = planner.plan_week(ings, dietary_filters=dietary_filters, indian_focus=indian_focus)
    st.session_state.day_plan = day
    st.session_state.week_plan = week
    st.session_state.plan_generated = True
    st.session_state.scroll_to_analysis = True
    st.session_state.day_plan_overrides = {}
    used = {r["title"] for r in day.values() if r is not None}
    st.session_state.used_titles_today = used
    invalidate_analysis_cache()
    invalidate_enrichment_cache()
    return day, week

def _get_default_quantity(item: str) -> str:
    s_item = item.lower()
    if any(s in s_item for s in ["salt", "pepper", "sugar", "spice", "cinnamon", "nutmeg", "paprika", "cumin", "powder", "oregano", "thyme", "basil", "coriander", "za atar", "vanilla", "cayenne", "seasoning"]):
        return "1 tsp"
    if any(l in s_item for l in ["oil", "vinegar", "sauce", "honey", "syrup", "butter", "paste", "mustard"]):
        return "2 tbsp"
    if any(m in s_item for m in ["milk", "cream", "water", "broth", "stock", "wine"]):
        return "1/2 cup"
    if any(p in s_item for p in ["chicken", "beef", "lamb", "pork", "salmon", "fish", "tofu", "paneer", "turkey", "duck"]):
        return "200g"
    if any(s in s_item for s in ["rice", "pasta", "noodles", "flour", "quinoa", "couscous", "oats", "bread", "dough"]):
        return "1 cup"
    if "egg" in s_item:
        return "2 whole"
    if any(v in s_item for v in ["onion", "tomato", "potato", "carrot", "bell pepper", "avocado", "apple", "lemon", "lime"]):
        return "1 whole"
    if any(v in s_item for v in ["spinach", "kale", "lettuce", "cabbage", "mushroom", "broccoli", "cauliflower", "green bean", "peas", "corn", "bean", "lentil", "chickpea"]):
        return "1/2 cup"
    if any(v in s_item for v in ["garlic", "ginger", "chili", "shallot"]):
        return "2 cloves/pieces"
    if any(c in s_item for c in ["cheese", "parmesan", "cheddar", "mozzarella", "feta", "ricotta"]):
        return "1/4 cup"
    return "To taste"


def update_ingredient_quantities(raw_text_or_list):
    """
    Parses the raw text/list to extract quantities and associates them with canonical names in st.session_state.ingredient_quantities.
    """
    if "ingredient_quantities" not in st.session_state:
        st.session_state.ingredient_quantities = {}
        
    normalizer = load_normalizer()
    if isinstance(raw_text_or_list, str):
        raw_list = normalizer.parse_text_input(raw_text_or_list)
    else:
        raw_list = raw_text_or_list
        
    # Get normalized mapping {canonical -> raw}
    normalized_map = normalizer.normalize_list(raw_list)
    
    for canonical, raw in normalized_map.items():
        qty, _ = normalizer.extract_quantity(raw)
        if qty:
            st.session_state.ingredient_quantities[canonical] = qty
        else:
            if canonical not in st.session_state.ingredient_quantities:
                st.session_state.ingredient_quantities[canonical] = ""


def _generate_steps(title: str, ingredients: str, instructions: str, cuisine: str) -> list[str]:
    """
    Generate rich, detailed step-by-step cooking instructions from recipe data.
    Falls back gracefully if ingredients or method info is sparse.
    """
    # Parse ingredient list
    ing_list = [i.strip().replace("_", " ") for i in ingredients.split("|") if i.strip()]
    
    def get_ing_with_qty(item_name: str) -> str:
        canonical_item = item_name.lower().replace(" ", "_")
        qty = st.session_state.get("ingredient_quantities", {}).get(canonical_item, "")
        if not qty:
            qty = _get_default_quantity(item_name)
        return f"{qty} {item_name}"

    # Separate protein, veg, spices, liquids
    proteins = [i for i in ing_list if any(p in i for p in [
        "chicken", "beef", "lamb", "pork", "fish", "salmon", "tuna", "cod", "shrimp",
        "prawn", "tofu", "paneer", "egg", "turkey", "duck", "crab", "lobster"
    ])]
    veggies  = [i for i in ing_list if any(v in i for v in [
        "tomato", "onion", "garlic", "pepper", "carrot", "celery", "spinach", "broccoli",
        "mushroom", "zucchini", "potato", "eggplant", "corn", "peas", "kale", "ginger",
        "leek", "cabbage", "cucumber", "squash", "pumpkin", "cauliflower", "green bean",
        "sweet potato", "capsicum", "spring onion", "scallion", "chili", "avocado"
    ])]
    spices   = [i for i in ing_list if any(s in i for s in [
        "salt", "pepper", "cumin", "turmeric", "paprika", "oregano", "thyme", "basil",
        "coriander", "cardamom", "cinnamon", "garam masala", "curry", "chili powder",
        "bay leaf", "star anise", "clove", "nutmeg", "saffron", "rosemary", "parsley"
    ])]
    liquids  = [i for i in ing_list if any(l in i for l in [
        "broth", "stock", "milk", "cream", "coconut milk", "water", "wine", "sauce",
        "oil", "vinegar", "lemon juice", "lime juice", "soy sauce", "tomato sauce"
    ])]
    starch   = [i for i in ing_list if any(s in i for s in [
        "rice", "pasta", "noodle", "bread", "flour", "potato", "quinoa", "couscous",
        "oat", "lentil", "chickpea", "bean", "barley"
    ])]

    # Detect cooking method from title / instructions
    raw = (title + " " + instructions).lower()
    method = "cook"
    if any(m in raw for m in ["grill", "bbq", "barbecue"]):  method = "grill"
    elif any(m in raw for m in ["stir-fr", "stir fr", "wok"]):  method = "stir-fry"
    elif any(m in raw for m in ["bake", "roast", "oven"]):   method = "roast"
    elif any(m in raw for m in ["brais", "simmer", "slow"]):  method = "braise"
    elif any(m in raw for m in ["steam"]):                    method = "steam"
    elif any(m in raw for m in ["fry", "pan-fr", "sauté", "saute"]): method = "pan-fry"
    elif any(m in raw for m in ["soup", "stew", "boil"]):    method = "simmer"
    elif any(m in raw for m in ["salad", "toss", "dress"]):  method = "toss"

    steps = []

    # Step 1 — Prep
    prep_items = []
    if veggies: prep_items.append(f"dice/prep the vegetables ({', '.join(get_ing_with_qty(v) for v in veggies[:3])})")
    if proteins: prep_items.append(f"cut the protein ({get_ing_with_qty(proteins[0])}) into bite-sized pieces")
    if spices: prep_items.append(f"measure out {', '.join(get_ing_with_qty(s) for s in spices[:3])}")
    if prep_items:
        steps.append(f"**Prep (5–10 min):** {'; '.join(prep_items).capitalize()}. Pat any proteins dry with paper towels for better browning.")

    # Step 2 — Base / aromatics
    aromatics = [i for i in ["garlic", "onion", "ginger", "shallot", "leek"] if any(i in v for v in ing_list)]
    oil = next((i for i in ing_list if "oil" in i or "butter" in i or "ghee" in i), "olive oil")
    oil_with_qty = get_ing_with_qty(oil)
    if aromatics:
        steps.append(
            f"**Build the base (3–4 min):** Heat {oil_with_qty} in a wide pan over medium-high heat. "
            f"Add {', '.join(get_ing_with_qty(a) for a in aromatics[:3])} and sauté until fragrant and lightly golden, about 2–3 minutes. "
            "Stir frequently to avoid burning."
        )

    # Step 3 — Protein
    if proteins:
        timing = {"grill": "4–5 min each side", "stir-fry": "3–4 min, tossing constantly",
                  "roast": "20–25 min at 200°C (400°F)", "braise": "5 min to sear, then covered",
                  "pan-fry": "4–5 min per side until golden", "steam": "10–12 min",
                  "simmer": "15–20 min in the liquid"}.get(method, "6–8 min until cooked through")
        steps.append(
            f"**Cook the protein ({timing}):** Add {', '.join(get_ing_with_qty(p) for p in proteins[:2])} and {method} for {timing}. "
            "Season generously with salt and pepper. "
            f"{'Ensure internal temperature reaches 75°C (165°F) for poultry.' if any('chicken' in p or 'turkey' in p for p in proteins) else 'Cook until no longer pink inside.'}"
        )

    # Step 4 — Vegetables
    if veggies:
        firm = [v for v in veggies if any(f in v for f in ["potato", "carrot", "broccoli", "cauliflower", "corn", "bean"])]
        soft = [v for v in veggies if v not in firm]
        if firm:
            steps.append(
                f"**Add firm vegetables (5–7 min):** Add {', '.join(get_ing_with_qty(f) for f in firm[:3])} to the pan. "
                "Stir well to coat with the cooking juices. Cook over medium heat until starting to soften."
            )
        if soft:
            steps.append(
                f"**Add tender vegetables (2–3 min):** Stir in {', '.join(get_ing_with_qty(s) for s in soft[:3])}. "
                "Cook briefly — these need just 2–3 minutes so they stay vibrant and slightly crisp."
            )

    # Step 5 — Liquid / sauce
    if liquids:
        liq = ', '.join(get_ing_with_qty(l) for l in liquids[:2])
        steps.append(
            f"**Add liquid & develop flavour (8–12 min):** Pour in {liq}. "
            f"Add {', '.join(get_ing_with_qty(s) for s in spices[:4]) if spices else 'salt and pepper'}. "
            f"{'Bring to a boil, then reduce heat and simmer' if method in ('simmer','braise') else 'Stir to deglaze the pan and combine'}. "
            "Taste and adjust seasoning."
        )

    # Step 6 — Starch
    if starch:
        starch_item = starch[0]
        cook_note = {
            "rice": "Rinse rice first; cook separately with a 1:2 rice-to-water ratio for 18 min.",
            "pasta": "Cook in well-salted boiling water until al dente, 8–10 min.",
            "noodle": "Soak or boil noodles per packet instructions, then drain.",
            "lentil": "Rinse lentils and add directly — they'll cook in 20–25 min.",
            "chickpea": "Use pre-cooked or canned chickpeas, drained and rinsed.",
        }.get(next((k for k in ["rice","pasta","noodle","lentil","chickpea"] if k in starch_item), ""), "Cook separately and serve alongside.")
        steps.append(f"**{starch_item.title()} ({get_ing_with_qty(starch_item)}):** {cook_note}")

    # Step 7 — Finish
    finish_herbs = [i for i in ing_list if any(h in i for h in ["parsley","cilantro","basil","mint","chive","spring onion","lemon","lime"])]
    finish_note = f"Garnish with fresh {', '.join(get_ing_with_qty(h) for h in finish_herbs[:2])} for brightness. " if finish_herbs else ""
    steps.append(
        f"**Plate & serve:** {finish_note}"
        f"Taste one final time and adjust salt, pepper, or acidity (a squeeze of lemon works wonders). "
        f"Serve {'hot' if method not in ('toss','salad') else 'at room temperature or slightly chilled'} and enjoy!"
    )

    return steps


def render_recipe_card(recipe: pd.Series, slot: str = "dinner", enrichment: dict = None):
    """Render a rich recipe card with inline step-by-step instructions."""
    title        = recipe.get("title", "Unknown")
    cuisine      = recipe.get("cuisine", "").title()
    tags         = [t.strip() for t in str(recipe.get("tags", "")).split(",") if t.strip()]
    instructions = str(recipe.get("instructions", ""))
    ingredients  = str(recipe.get("ingredients", ""))
    cals         = int(recipe.get("calories", 0))
    protein      = float(recipe.get("protein", 0))
    carbs        = float(recipe.get("carbs", 0))
    fat          = float(recipe.get("fat", 0))
    coverage     = round(float(recipe.get("coverage", 0)) * 100, 0)
    missing      = recipe.get("missing_ingredients", [])
    if isinstance(missing, str):
        missing = [m.strip() for m in missing.split(",") if m.strip()]

    # Enrichment data (difficulty, prep time, why recommended)
    diff_level = enrichment.get("difficulty", 2) if enrichment else 2
    diff_notes = enrichment.get("difficulty_notes", "") if enrichment else ""
    prep_time  = enrichment.get("prep_time_min", 30) if enrichment else 30
    why_text   = enrichment.get("why_recommended", "") if enrichment else ""

    # Difficulty stars
    stars = "★" * diff_level + "☆" * (5 - diff_level)

    tag_html = "".join(
        f'<span class="recipe-badge">{t}</span>' for t in tags[:5]
    )
    missing_html = ""
    if missing:
        pills = "".join(
            f'<span class="chip-missing">{m.replace("_", " ")}</span>'
            for m in missing[:6]
        )
        missing_html = f'<div style="margin-top:0.6rem;">🛒 <span style="color:#94a3b8;font-size:0.78rem;">Still need: </span>{pills}</div>'

    why_html = f'<div class="why-recommended">{why_text}</div>' if why_text else ""

    # Render the card header
    st.markdown(
        f"""
<div class="recipe-card {slot}">
  <div class="recipe-title">{title}</div>
  <div class="recipe-meta-row">
    <span style="color:#94a3b8;font-size:0.82rem;">🌍 {cuisine}</span>
    <span style="color:#94a3b8;font-size:0.82rem;">📊 {coverage:.0f}% match</span>
    <span class="time-pill">⏱ {prep_time} MIN</span>
    <span class="difficulty-badge difficulty-{diff_level}">{stars} <span style="margin-left:0.2rem;">{diff_notes}</span></span>
  </div>
  <div>{tag_html}</div>
  <div class="nutrition-row">
    <div class="nutrition-pill"><span class="value">{cals}</span><span class="label">kcal</span></div>
    <div class="nutrition-pill"><span class="value">{protein:.0f}g</span><span class="label">protein</span></div>
    <div class="nutrition-pill"><span class="value">{carbs:.0f}g</span><span class="label">carbs</span></div>
    <div class="nutrition-pill"><span class="value">{fat:.0f}g</span><span class="label">fat</span></div>
  </div>
  <div class="coverage-bar-outer">
    <div class="coverage-bar-inner" data-fill="{coverage}%"></div>
  </div>
  {missing_html}
  {why_html}
</div>
""",
        unsafe_allow_html=True,
    )

    def _get_quantity(item: str) -> str:
        canonical_item = item.lower().replace(" ", "_")
        user_qty = st.session_state.get("ingredient_quantities", {}).get(canonical_item, "")
        if user_qty:
            return user_qty
        return _get_default_quantity(item)

    # ── Render Ingredients and Instructions in separate tabs ───────────────
    # Parse ingredients to show checkmarks/crosses
    full_ing_list = [i.strip().replace("_", " ").capitalize() for i in ingredients.split("|") if i.strip()]
    missing_normalized = [m.replace("_", " ").capitalize() for m in missing]

    tab_ing, tab_inst = st.tabs(["🛒 Ingredients", "📋 Instructions"])

    with tab_ing:
        ing_html = '<div style="padding: 0.5rem 0.5rem 1rem 0.5rem;">'
        ing_html += '<div style="font-size:0.85rem; color:#94a3b8; margin-bottom: 0.8rem;">What you need for this recipe:</div>'
        
        for item in full_ing_list:
            qty = _get_quantity(item)
            if item in missing_normalized:
                ing_html += f'<div style="margin-bottom:0.4rem; font-size:0.9rem; display:flex; justify-content:space-between;"><span style="color:#cbd5e1;"><span style="color:#ef4444; margin-right:0.4rem;">⭕</span>{item} <span style="color:#64748b; font-size:0.75rem;">(missing)</span></span><span style="color:#94a3b8; font-weight:bold;">{qty}</span></div>'
            else:
                ing_html += f'<div style="margin-bottom:0.4rem; font-size:0.9rem; display:flex; justify-content:space-between;"><span style="color:#F1F5F9;"><span style="color:#5be38d; margin-right:0.4rem;">✅</span>{item}</span><span style="color:#94a3b8; font-weight:bold;">{qty}</span></div>'
        ing_html += '</div>'
        st.markdown(ing_html, unsafe_allow_html=True)

    with tab_inst:
        steps = _generate_steps(title, ingredients, instructions, cuisine)
        
        # Action row for the read aloud button
        plain_text = " ".join([s.replace("**", "") for s in steps])
        components.html(get_read_aloud_button_html(plain_text, lang_code="en-IN"), height=45)

        st.markdown('<div style="padding: 0.5rem;">', unsafe_allow_html=True)
        for i, step in enumerate(steps, 1):
            if ":**" in step:
                label, body = step.split(":**", 1)
                label = label.replace("**", "").strip()
                body  = body.strip()
            else:
                label, body = f"Step {i}", step

            st.markdown(
                f"""
    <div class="instruction-step" style="--step-index:{i - 1};">
      <div class="instruction-step-num">{i}</div>
      <div>
        <span style="font-weight:700;color:#5be38d;font-size:0.88rem;">{label}: </span>
        <span style="color:#94a3b8;font-size:0.88rem;line-height:1.6;">{body}</span>
      </div>
    </div>
    """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)



def render_ingredient_chips(ingredients: list):
    """Render ingredients as styled chips with quantities if available."""
    if not ingredients:
        st.markdown(
            '<div class="chip-container"><span style="color:#64748b;font-size:0.85rem;">No ingredients added yet…</span></div>',
            unsafe_allow_html=True,
        )
        return
    chips = []
    quantities = st.session_state.get("ingredient_quantities", {})
    for idx, i in enumerate(ingredients):
        qty = quantities.get(i, "")
        display_name = i.replace("_", " ")
        if qty:
            display_text = f"{qty} {display_name}"
        else:
            display_text = display_name
        chips.append(f'<span class="chip" style="--chip-index:{idx}">{display_text}</span>')
    st.markdown(f'<div class="chip-container">{"".join(chips)}</div>', unsafe_allow_html=True)


def render_nutrition_chart(nutrition: dict, title: str = "Macro Breakdown"):
    """Donut chart for macros."""
    labels = ["Carbs", "Protein", "Fat"]
    values = [
        nutrition.get("carbs", 0) * 4,
        nutrition.get("protein", 0) * 4,
        nutrition.get("fat", 0) * 9,
    ]
    colors = ["#3b82f6", "#5be38d", "#eab308"]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=15)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8")),
        margin=dict(t=40, b=20, l=20, r=20),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})  # noqa: deprecated-ok


def render_weekly_bar_chart(weekly_nutrition: dict):
    """Stacked bar chart for weekly calorie distribution."""
    days = [d for d in DAYS if d in weekly_nutrition]
    cals = [weekly_nutrition[d]["calories"] for d in days]
    protein_cals = [weekly_nutrition[d]["protein"] * 4 for d in days]
    carb_cals = [weekly_nutrition[d]["carbs"] * 4 for d in days]
    fat_cals = [weekly_nutrition[d]["fat"] * 9 for d in days]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Carbs", x=days, y=carb_cals, marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="Protein", x=days, y=protein_cals, marker_color="#5be38d"))
    fig.add_trace(go.Bar(name="Fat", x=days, y=fat_cals, marker_color="#eab308"))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", title="Calories (kcal)"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})  # noqa: deprecated-ok


# ── Session state defaults ─────────────────────────────────────────────────────

def init_state():
    defaults = {
        "canonical_ingredients": [],
        "raw_ingredient_text": "",
        "day_plan": None,
        "week_plan": None,
        "detected_vision": [],
        "vision_strings": [],
        "plan_generated": False,
        "calorie_goal": 2000,
        "day_plan_overrides": {},    # {slot: recipe_row} — per-slot regenerations
        "used_titles_today": set(),  # tracks all titles shown today to avoid repeats
        "chat_messages": [
            {"role": "assistant", "content": "👋 Hello! I'm your AI Chef. You can tell me what ingredients you have, ask me to generate a meal plan, or ask any cooking questions!"}
        ],
        "diet_profile": {
            "age": 30,
            "gender": "Male",
            "weight": 70.0,
            "height": 175.0,
            "activity": "Sedentary (Little/no exercise)",
            "goal": "Maintain Weight",
            "macro_preset": "Balanced (50/25/25)"
        },
        "diet_targets": {
            "bmr": 1618.8,
            "tdee": 1942.6,
            "target_calories": 2000,
            "macro_targets": {"carbs": 250.0, "protein": 125.0, "fat": 56.0},
            "macro_percentages": {"carbs": 50, "protein": 25, "fat": 25}
        },
        "diet_chat_messages": [
            {"role": "assistant", "content": "👋 Welcome to your Diet & Nutrition Assistant! I am your AI Dietitian and Health Coach. Enter your details to calculate your target calorie and macronutrient limits, then ask me anything about your diet plan!"}
        ],
        "leftover_items": [],
        "leftover_raw_text": "",
        "leftover_results": None,
        "budget_limit": 15.0,
        "barcode_data": None,
        "custom_recipe": None,
        "indian_focus": True,
        "filter_vegetarian": False,
        "filter_vegan": False,
        "filter_gluten_free": False,
        "ingredient_quantities": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    init_state()
    inject_ui_enhancements()

    # ── Sidebar — must come BEFORE load_vision so key is set first ─────────────
    # ── Floating Settings Popover ──────────────────────────────────────────────
    st.markdown('<div class="floating-settings">', unsafe_allow_html=True)
    with st.popover("⚙️ Settings"):
        st.markdown("## ⚙️ Preferences")
        st.markdown("---")

        st.markdown("**🥦 Dietary Filters**")
        
        def on_filter_change():
            st.session_state.plan_generated = False
            invalidate_analysis_cache()
            
        vegetarian = st.checkbox("Vegetarian", key="filter_vegetarian", on_change=on_filter_change)
        vegan = st.checkbox("Vegan", key="filter_vegan", on_change=on_filter_change)
        gluten_free = st.checkbox("Gluten-Free", key="filter_gluten_free", on_change=on_filter_change)
        indian_focus = st.checkbox("Focus on Indian Recipes", key="indian_focus", on_change=on_filter_change)

        dietary_filters = []
        if st.session_state.filter_vegetarian:
            dietary_filters.append("vegetarian")
        if st.session_state.filter_vegan:
            dietary_filters.append("vegan")
        if st.session_state.filter_gluten_free:
            dietary_filters.append("gluten-free")

        st.markdown("---")
        st.markdown("**🎯 Daily Calorie Goal**")
        calorie_goal = st.slider(
            "Target kcal/day",
            min_value=1200,
            max_value=4000,
            value=st.session_state.calorie_goal,
            step=50,
            help="We'll show a progress bar on Today's Plan comparing actual vs target.",
        )
        st.session_state.calorie_goal = calorie_goal

        # Macro split preset
        macro_preset = st.selectbox(
            "Macro Split Preset",
            ["Balanced (50/25/25)", "High Protein (40/35/25)", "Low Carb (25/35/40)", "Custom"],
            index=0,
            help="Carbs / Protein / Fat target percentages",
        )
        macro_targets = {
            "Balanced (50/25/25)":    {"carbs": 50, "protein": 25, "fat": 25},
            "High Protein (40/35/25)": {"carbs": 40, "protein": 35, "fat": 25},
            "Low Carb (25/35/40)":    {"carbs": 25, "protein": 35, "fat": 40},
        }.get(macro_preset, {"carbs": 50, "protein": 25, "fat": 25})
        st.session_state["macro_targets"] = macro_targets

        st.markdown("---")
        st.markdown("**🔑 Vision API Key**")
        st.caption("Add ONE of the keys below (Groq is recommended — 100% free)")

        groq_input = st.text_input(
            "🟢 Groq API Key (free · fast)",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Free at console.groq.com — no credit card needed. Key starts with gsk_",
            placeholder="gsk_…",
        )
        gemini_input = st.text_input(
            "🔵 Gemini API Key (free tier)",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="Free at aistudio.google.com/app/apikey — key starts with AIzaSy",
            placeholder="AIzaSy…",
        )
        openai_input = st.text_input(
            "⚫ OpenAI API Key (GPT-4o)",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Requires paid API access at platform.openai.com. Key starts with sk-",
            placeholder="sk-…",
        )

        if groq_input:
            os.environ["GROQ_API_KEY"] = groq_input
        if gemini_input:
            os.environ["GEMINI_API_KEY"] = gemini_input
        if openai_input:
            os.environ["OPENAI_API_KEY"] = openai_input

        # Show status
        if groq_input or os.getenv("GROQ_API_KEY", ""):
            st.success("✅ Groq active — photo detection ready")
        elif gemini_input or os.getenv("GEMINI_API_KEY", ""):
            st.success("✅ Gemini active — photo detection ready")
        elif openai_input or os.getenv("OPENAI_API_KEY", ""):
            st.success("✅ OpenAI active — photo detection ready")
        else:
            st.info("ℹ️ No key set — get a free Groq key at console.groq.com")

        st.markdown("---")
        st.markdown("**🤖 AI Provider Override**")
        ai_override = st.radio(
            "Select AI Engine",
            options=["Auto (Groq first)", "OpenAI", "Gemini", "Groq"],
            index=0,
            help="Forces the use of a specific AI model. Falls back to auto if key is missing."
        )
        provider_val = "auto"
        if ai_override == "Gemini":
            provider_val = "gemini"
        elif ai_override == "Groq":
            provider_val = "groq"
        elif ai_override == "OpenAI":
            provider_val = "openai"

        st.markdown("---")
        st.markdown("**💸 Daily Meal Budget**")
        budget_limit = st.slider(
            "Budget per day (USD)",
            min_value=5.0,
            max_value=50.0,
            value=st.session_state.budget_limit,
            step=0.5,
            help="We'll warn you if today's meal plan exceeds this cost.",
        )
        st.session_state.budget_limit = budget_limit

        st.markdown("---")
        if st.session_state.canonical_ingredients:
            n = len(st.session_state.canonical_ingredients)
            st.markdown(
                f"""
<div class="stat-card">
  <div class="stat-value">{n}</div>
  <div class="stat-label">Ingredients Loaded</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown(
            '<div style="color:#64748b;font-size:0.75rem;text-align:center;">Smart Meal Planner v2.0<br>Powered by Gemini + NLP + AI</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # Load modules (vision auto-picks Groq → Gemini based on env vars)
    normalizer = load_normalizer()
    matcher = load_matcher()
    planner = load_planner(matcher)
    nutrition = load_nutrition()
    diet_assistant = load_diet_assistant()
    budget_optimizer = load_budget_optimizer(st.session_state.budget_limit)
    rec_engine = load_recommendation_engine()
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    vision = load_vision(gemini_key=gemini_key, groq_key=groq_key, openai_key=openai_key, provider=provider_val)
    intent_parser = load_intent_parser(gemini_key=gemini_key, groq_key=groq_key, openai_key=openai_key, provider=provider_val)

    # ── Hero Banner ────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="hero-banner">
  <div class="hero-emoji">🍱</div>
  <div>
    <div class="hero-title">Smart Meal Planner</div>
    <div class="hero-sub">AI-powered meal planning · Nutrition coaching · Budget optimization · Leftover smart matching</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Main Navigation ────────────────────────────────────────────────
    tab_ingredients, tab_today, tab_weekly, tab_nutrition, tab_diet, tab_chat = st.tabs(
        ["🥕 Ingredients", "🍽️ Today's Plan", "📅 Weekly Plan", "📊 Nutrition", "🥗 Diet Assistant", "💬 AI Chef"]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — INGREDIENTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ingredients:
        col_text, col_photo = st.columns([1, 1], gap="large")

        with col_text:
            st.markdown(
                '<div class="section-header">✏️ Add Ingredients (Text & Voice)</div>',
                unsafe_allow_html=True,
            )

            tab_type, tab_speak = st.tabs(["⌨️ Type", "🎤 Speak"])

            # Language selector lives outside the tabs so selected_lang is always in scope
            selected_lang = st.selectbox("Language", options=list(LANGUAGES.keys()), index=0, key="lang_selector")

            with tab_speak:
                # Primary: Native Browser Web Speech (Real-time, handles translations)
                components.html(get_voice_widget_html(LANGUAGES[selected_lang], "main_voice"), height=220)
                
                # Fallback: Python SpeechRecognition for devices where Web Speech isn't supported
                with st.expander("Alternative: Record Audio Clip"):
                    audio_bytes = st.audio_input("Record audio", key="audio_fallback")
                    # Use a session state flag so we don't transcribe the same audio twice on re-runs
                    if audio_bytes and st.session_state.get("last_audio_hash") != hash(audio_bytes.getvalue()):
                        st.session_state["last_audio_hash"] = hash(audio_bytes.getvalue())
                        with custom_spinner("Transcribing..."):
                            text, err = transcribe_audio(audio_bytes.getvalue(), LANGUAGES[selected_lang])
                            if text:
                                if st.session_state.raw_ingredient_text:
                                    st.session_state.raw_ingredient_text += "\n" + text
                                else:
                                    st.session_state.raw_ingredient_text = text
                                st.success(f"Heard: {text}")
                                st.rerun()
                            elif err:
                                st.error(err)

            with tab_type:
                st.info("💡 You can paste copied voice text here, or type manually.")
                raw_text = st.text_area(
                    "Enter ingredients",
                    value=st.session_state.raw_ingredient_text,
                    height=180,
                    placeholder="e.g.\n2 cups of flour\n3 tomatoes\nchicken breast\nonion, garlic, butter",
                    help="Separate ingredients by comma, newline, or semicolon. Quantities and units are handled automatically.",
                    label_visibility="collapsed",
                )
                st.session_state.raw_ingredient_text = raw_text

            if st.button("Submit"):
                if raw_text.strip():
                    with custom_spinner("🤖 Understanding your request..."):
                        parsed = intent_parser.parse_intent(raw_text)
                        
                    intent = parsed.get("intent", "ADD_INGREDIENTS")
                    extracted_ing = _extract_ingredients_from_text(raw_text, parsed, normalizer)
                    chat_response = parsed.get("chat_response", "")

                    speak_text = ""
                    
                    if intent == "CHAT":
                        # Add user message and assistant message to chat history
                        st.session_state.chat_messages.append({"role": "user", "content": raw_text})
                        if chat_response:
                            st.session_state.chat_messages.append({"role": "assistant", "content": chat_response})
                        st.success("✅ AI Chef responded! Switch to the 💬 AI Chef tab to see the chat.")
                        speak_text += "I've answered your question. Check the AI Chef tab."
                        
                    elif extracted_ing:
                        # Normalize the extracted ingredients
                        canonical = normalizer.get_ingredient_set(", ".join(extracted_ing))
                        if canonical:
                            st.session_state.canonical_ingredients = canonical
                            update_ingredient_quantities(extracted_ing)
                            invalidate_analysis_cache()
                            st.success(f"✅ Extracted ingredients: {', '.join(canonical)}")
                            speak_text += f"I have added {len(canonical)} ingredients to your list. "
                        else:
                            st.warning("Could not clearly map ingredients. Please check spelling.")
                            speak_text += "I couldn't understand those ingredients. "
                    elif intent == "ADD_INGREDIENTS":
                        st.warning("Could not extract any ingredients from your message.")
                        speak_text += "I didn't catch any ingredients in your message. "

                    if intent == "GENERATE_MEAL" and st.session_state.canonical_ingredients:
                        with custom_spinner("🚀 Generating meal plans based on your request..."):
                            _apply_meal_plan(
                                st.session_state.canonical_ingredients,
                                planner,
                                dietary_filters,
                                st.session_state.indian_focus,
                            )
                        st.success("✅ Plans generated! Switch to the 'Today\'s Plan' or 'Weekly Plan' tab.")
                        speak_text += "Your meal plan is ready! Switch to the plan tabs to see it."
                    elif intent == "GENERATE_MEAL":
                        st.warning("Please tell me what ingredients you have before generating a meal!")
                        speak_text += "Please tell me what ingredients you have first."

                    if speak_text:
                        components.html(speak_text_html(speak_text, lang_code=LANGUAGES.get(selected_lang, "en-US")), height=0)
                        
                else:
                    st.warning("Please enter some text or use the voice assistant first.")

        with col_photo:
            st.markdown(
                '<div class="section-header">📸 Or Upload a Photo</div>',
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Food photo",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
                help="Upload a photo of your fridge, pantry, or ingredients.",
            )
            if uploaded:
                if "_flash_msg" in st.session_state:
                    st.success(st.session_state._flash_msg)
                    del st.session_state._flash_msg
                    
                st.image(uploaded, use_container_width=True)
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🤖 Detect Ingredients", use_container_width=True):
                        if not vision.is_available:
                            st.error("Vision API not configured.")
                        else:
                            with custom_spinner("🔍 Analyzing photo…"):
                                img_bytes = uploaded.read()
                                mime = uploaded.type or "image/jpeg"
                                detected, err = vision.detect_ingredients(img_bytes, mime)
                            if err:
                                st.error(f"Vision error: {err}")
                            elif detected:
                                vision_strings = vision.detected_to_strings(detected)
                                st.session_state.vision_strings = vision_strings
                                st.session_state.raw_ingredient_text = "\n".join(vision_strings)
                                canonical = normalizer.get_ingredient_set("\n".join(vision_strings))
                                st.session_state.canonical_ingredients = canonical
                                update_ingredient_quantities(vision_strings)
                                st.session_state.plan_generated = False
                                invalidate_analysis_cache()
                                
                                with custom_spinner("🚀 Generating meal plans based on your photo...", keep_alive=True):
                                    _apply_meal_plan(
                                        st.session_state.canonical_ingredients,
                                        planner,
                                        dietary_filters,
                                        st.session_state.indian_focus,
                                    )
                                
                                st.session_state._flash_msg = f"✅ Detected {len(detected)} ingredients and generated your meal plan!"
                                st.rerun()
                            else:
                                st.warning("No food items detected.")
                
                with col_btn2:
                    if st.button("🏷️ Scan Barcode/Label", use_container_width=True):
                        if not vision.is_available:
                            st.error("Vision API not configured.")
                        else:
                            with custom_spinner("🔍 Scanning nutrition label…"):
                                img_bytes = uploaded.read()
                                mime = uploaded.type or "image/jpeg"
                                data, err = vision.scan_barcode_or_label(img_bytes, mime)
                            if err:
                                st.error(f"Scan error: {err}")
                            elif data:
                                st.session_state.barcode_data = data
                                st.success(f"✅ Scanned: {data.get('product_name', 'Unknown Product')}")
                                
            if st.session_state.barcode_data:
                d = st.session_state.barcode_data
                st.markdown(f"""
                <div style="background:var(--card-bg); padding:1rem; border-radius:12px; margin-top:10px;">
                    <h4>📊 {d.get('product_name')} {f"({d.get('brand')})" if d.get('brand') else ""}</h4>
                    <p><b>Calories:</b> {d.get('calories_per_serving')} kcal per {d.get('serving_size')}</p>
                    <p><b>Macros:</b> {d.get('protein_g')}g Protein | {d.get('carbs_g')}g Carbs | {d.get('fat_g')}g Fat</p>
                    <p><b>Ingredients:</b> {", ".join(d.get('ingredient_list', []))}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("➕ Add to ingredients list", use_container_width=True):
                    if d.get("product_name"):
                        st.session_state.canonical_ingredients.append(d.get("product_name").lower().replace(" ", "_"))
                        st.session_state.canonical_ingredients = list(set(st.session_state.canonical_ingredients))
                        st.success("Added to ingredients!")

        # ── Ingredient review ──────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header">🏷️ Detected Ingredients (editable)</div>',
            unsafe_allow_html=True,
        )

        ingredients = st.session_state.canonical_ingredients
        render_ingredient_chips(ingredients)

        if ingredients:
            # Allow user to remove / add
            with st.expander("✏️ Edit ingredient list"):
                lines = []
                quantities = st.session_state.get("ingredient_quantities", {})
                for i in ingredients:
                    qty = quantities.get(i, "")
                    display_name = i.replace("_", " ")
                    if qty:
                        lines.append(f"{qty} {display_name}")
                    else:
                        lines.append(display_name)
                
                edited = st.text_area(
                    "One ingredient per line (e.g. 500g chicken)",
                    value="\n".join(lines),
                    height=150,
                    label_visibility="collapsed",
                )
                if st.button("💾 Apply Changes"):
                    raw_lines = [l.strip() for l in edited.splitlines() if l.strip()]
                    normalized_map = normalizer.normalize_list(raw_lines)
                    st.session_state.canonical_ingredients = sorted(normalized_map.keys())
                    update_ingredient_quantities(raw_lines)
                    st.session_state.plan_generated = False
                    invalidate_analysis_cache()
                    st.success("Updated!")
                    st.rerun()



            st.markdown("---")

            # ── Confidence Analysis ───────────────────────────────────────────
            analysis = get_cached_analysis(
                ingredients,
                st.session_state.day_plan,
                st.session_state.week_plan,
                rec_engine,
                matcher,
            )
            conf = analysis["confidence"]
            score = conf["score"]
            grade = conf["grade"]

            # Color for ring — KitchenIQ palette
            if score >= 70:
                ring_color = "#5be38d"
                grade_color = "#5be38d"
            elif score >= 40:
                ring_color = "#eab308"
                grade_color = "#eab308"
            else:
                ring_color = "#ef4444"
                grade_color = "#ef4444"

            circumference = 2 * 3.14159 * 52
            dash_offset = circumference * (1 - score / 100)

            # Food group pills
            group_pills_html = ""
            group_icons = {"protein": "🥩", "carbs": "🌾", "vegetables": "🥬", "fats": "🫒", "fruits": "🍎"}
            for grp, present in conf.get("food_groups", {}).items():
                cls = "present" if present else "missing"
                icon = group_icons.get(grp, "")
                label = "✓" if present else "✗"
                group_pills_html += f'<span class="food-group-pill {cls}">{icon} {grp.capitalize()} {label}</span>'

            # Tips
            tips_html = ""
            for tip in conf.get("tips", [])[:3]:
                tips_html += f'<div class="confidence-tip"><span>💡</span><span>{tip}</span></div>'

            st.markdown(f"""
<div class="confidence-ring-container">
  <div class="confidence-ring">
    <svg viewBox="0 0 120 120">
      <circle class="ring-bg" cx="60" cy="60" r="52"/>
      <circle class="ring-fill" cx="60" cy="60" r="52"
              stroke="{ring_color}"
              stroke-dasharray="{circumference:.1f}"
              stroke-dashoffset="{dash_offset:.1f}"/>
    </svg>
    <div class="confidence-score-text">
      <div class="score">{score}</div>
      <div class="grade" style="color:{grade_color}">{grade}</div>
    </div>
  </div>
  <div class="confidence-details">
    <h3>🎯 Ingredient Confidence</h3>
    <div class="food-group-grid">{group_pills_html}</div>
    <div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.6rem;">
      🌍 Matches recipes across <strong style="color:var(--text-primary)">{conf.get('cuisine_reach', 0)}</strong> cuisines
      &nbsp;·&nbsp; 🧂 Pantry staples: <strong style="color:var(--text-primary)">{conf.get('staple_coverage', 0):.0f}%</strong>
    </div>
    {tips_html}
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Kitchen Optimization Tips ─────────────────────────────────────
            staple_unlocks = analysis.get("staple_unlocks", [])
            seasonal_swaps = analysis.get("seasonal_swaps", [])
            batch_cooking = analysis.get("batch_cooking", [])
            season = analysis.get("current_season", "")

            has_tips = staple_unlocks or seasonal_swaps or batch_cooking
            if has_tips:
                st.markdown(f"""<div class="kitchen-tips-card">
  <h4>🧠 Kitchen Optimization Tips — {season.capitalize()}</h4>""", unsafe_allow_html=True)

                # Staple unlocks
                if staple_unlocks:
                    st.markdown('<div class="unlock-card" style="padding-bottom: 8px;"><h4>🔓 Add These to Unlock More Recipes</h4>', unsafe_allow_html=True)
                    
                    def _add_unlock_ing(ing_name):
                        current = set(st.session_state.canonical_ingredients or [])
                        if ing_name.lower() not in current:
                            current.add(ing_name.lower())
                            st.session_state.canonical_ingredients = sorted(list(current))
                            update_ingredient_quantities([ing_name.lower()])
                            st.session_state.plan_generated = False
                            invalidate_analysis_cache()
                    
                    for idx, item in enumerate(staple_unlocks[:5]):
                        recipes_str = ", ".join(item["top_recipes"][:2])
                        col_text, col_btn = st.columns([5, 1])
                        with col_text:
                            st.markdown(f"""
<div class="unlock-item" style="border: none; padding: 4px 0;">
  <div>
    <span class="unlock-name">{item['ingredient'].capitalize()}</span>
    <span style="font-size:0.72rem;color:var(--text-muted);margin-left:0.5rem;">e.g. {recipes_str}</span>
  </div>
  <span class="unlock-count">+{item['unlocks']} recipes</span>
</div>""", unsafe_allow_html=True)
                        with col_btn:
                            st.markdown("<div style='padding-top:0.2rem'></div>", unsafe_allow_html=True)
                            st.button("➕ Add", key=f"add_unlock_{item['ingredient']}_{idx}", on_click=_add_unlock_ing, args=(item['ingredient'],), use_container_width=True)
                        
                        if idx < len(staple_unlocks[:5]) - 1:
                            st.markdown("<hr style='margin: 4px 0; border-color: var(--card-border);'>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Seasonal swaps
                if seasonal_swaps:
                    for sw in seasonal_swaps[:3]:
                        st.markdown(f"""
<div class="tip-item">
  <span class="tip-icon">🌿</span>
  <div class="tip-content">
    <div class="tip-title">Swap {sw['original'].capitalize()} → {sw['swap'].capitalize()}</div>
    <div class="tip-desc">{sw['reason']}</div>
  </div>
</div>""", unsafe_allow_html=True)

                # Batch cooking
                if batch_cooking:
                    for bc in batch_cooking[:3]:
                        meals_str = ", ".join(bc["meals"][:3])
                        st.markdown(f"""
<div class="tip-item">
  <span class="tip-icon">🍳</span>
  <div class="tip-content">
    <div class="tip-title">Batch prep: {bc['ingredient'].capitalize()}</div>
    <div class="tip-desc">Used in {bc['used_in']} recipes ({meals_str}). Prep once, use all week.</div>
  </div>
</div>""", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("")
            if st.button(
                "🚀 Generate Meal Plan",
                use_container_width=True,
                type="primary",
            ):
                with custom_spinner("AI Chef is cooking up your meal plan..."):
                    ings = st.session_state.canonical_ingredients
                    _apply_meal_plan(ings, planner, dietary_filters, st.session_state.indian_focus)
                
                st.balloons()
                st.success("✅ Plans generated! Switch to the Today or Weekly tab.")
        else:
            st.markdown(
                '<div class="empty-state"><div class="icon">🥕</div><div class="msg">Add some ingredients above to get started!</div></div>',
                unsafe_allow_html=True,
            )

        if st.session_state.get("scroll_to_analysis"):
            st.session_state.scroll_to_analysis = False
            components.html("""
            <script>
                setTimeout(() => {
                    function ultraSmoothScroll() {
                        const targetElement = window.parent.document.querySelector('.confidence-ring-container');
                        const scrollContainer = window.parent.document.querySelector('.stApp') || window.parent.document.querySelector('.main') || window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                        
                        if (!targetElement || !scrollContainer) return;
                        
                        const startY = scrollContainer.scrollTop;
                        const targetY = targetElement.getBoundingClientRect().top + scrollContainer.scrollTop - scrollContainer.getBoundingClientRect().top - 40;
                        const distance = targetY - startY;
                        const duration = 1500; // 1.5 seconds for buttery smoothness
                        let start = null;

                        function step(timestamp) {
                            if (!start) start = timestamp;
                            const progress = timestamp - start;
                            const t = Math.min(progress / duration, 1);
                            
                            // easeInOutQuart curve for extreme smoothness
                            const ease = t < 0.5 ? 8 * Math.pow(t, 4) : 1 - Math.pow(-2 * t + 2, 4) / 2;
                            
                            scrollContainer.scrollTo(0, startY + (distance * ease));
                            
                            if (progress < duration) {
                                window.parent.requestAnimationFrame(step);
                            }
                        }
                        window.parent.requestAnimationFrame(step);
                    }
                    ultraSmoothScroll();
                }, 1000);
            </script>
            """, height=0)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — TODAY'S PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_today:
        if not st.session_state.plan_generated or st.session_state.day_plan is None:
            st.markdown(
                '<div class="empty-state"><div class="icon">🍽️</div><div class="msg">Generate a meal plan from the Ingredients tab first.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            day_plan = get_effective_day_plan()
            day_nutrition = nutrition.aggregate_day(day_plan)
            goal_kcal = st.session_state.calorie_goal
            actual_kcal = day_nutrition["calories"]
            pct = actual_kcal / goal_kcal * 100 if goal_kcal else 0

            # ── Calorie Goal Progress Bar ─────────────────────────────────────
            if pct <= 110:
                bar_color = "linear-gradient(90deg, #5be38d, #3db872)"
                status_icon = "✅"
                status_label = "On target"
                status_color = "#5be38d"
            elif pct <= 125:
                bar_color = "linear-gradient(90deg, #eab308, #ca8a04)"
                status_icon = "⚠️"
                status_label = "Slightly over"
                status_color = "#eab308"
            else:
                bar_color = "linear-gradient(90deg, #ef4444, #dc2626)"
                status_icon = "🔴"
                status_label = "Over target"
                status_color = "#ef4444"

            bar_width = min(pct, 100)

            # Macro progress vs targets
            macro_targets = st.session_state.get("macro_targets", {"carbs": 50, "protein": 25, "fat": 25})
            total_macro_kcal = (
                day_nutrition["carbs"] * 4
                + day_nutrition["protein"] * 4
                + day_nutrition["fat"] * 9
            ) or 1
            actual_carbs_pct  = round(day_nutrition["carbs"]   * 4 / total_macro_kcal * 100)
            actual_protein_pct = round(day_nutrition["protein"] * 4 / total_macro_kcal * 100)
            actual_fat_pct    = round(day_nutrition["fat"]      * 9 / total_macro_kcal * 100)

            st.markdown(
                f"""
<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);
            padding:1.4rem 1.6rem;margin-bottom:1.5rem;box-shadow:var(--shadow-card);">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
    <div>
      <span style="font-size:1.05rem;font-weight:700;color:var(--text-primary);">🎯 Daily Calorie Goal</span>
      <span style="margin-left:0.8rem;font-size:0.82rem;color:{status_color};font-weight:600;">{status_icon} {status_label}</span>
    </div>
    <div style="font-size:0.92rem;color:var(--text-secondary);">
      <span style="font-weight:700;color:{status_color};">{int(actual_kcal)}</span> / {goal_kcal} kcal
      &nbsp;<span style="color:var(--text-muted);font-size:0.78rem;">({pct:.0f}%)</span>
    </div>
  </div>
  <div style="background:rgba(26,30,36,0.8);border-radius:6px;height:10px;overflow:hidden;margin-bottom:1rem;">
    <div class="progress-bar-fill" data-fill="{bar_width:.1f}%" style="width:0;height:100%;border-radius:6px;
                background:{bar_color};
                box-shadow:0 0 10px rgba(91,227,141,0.3);"></div>
  </div>
  <div style="display:flex;gap:1rem;flex-wrap:wrap;">
    <div style="flex:1;min-width:80px;">
      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">🌾 Carbs</div>
      <div style="font-size:0.88rem;font-weight:600;color:#3b82f6;">{actual_carbs_pct}% <span style="color:var(--text-muted);font-weight:400;font-size:0.75rem;">/ target {macro_targets['carbs']}%</span></div>
      <div style="background:rgba(26,30,36,0.8);border-radius:3px;height:4px;margin-top:0.3rem;">
        <div style="width:{min(actual_carbs_pct/macro_targets['carbs']*100,100):.0f}%;height:100%;border-radius:3px;background:#3b82f6;"></div>
      </div>
    </div>
    <div style="flex:1;min-width:80px;">
      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">💪 Protein</div>
      <div style="font-size:0.88rem;font-weight:600;color:#5be38d;">{actual_protein_pct}% <span style="color:var(--text-muted);font-weight:400;font-size:0.75rem;">/ target {macro_targets['protein']}%</span></div>
      <div style="background:rgba(26,30,36,0.8);border-radius:3px;height:4px;margin-top:0.3rem;">
        <div style="width:{min(actual_protein_pct/macro_targets['protein']*100,100):.0f}%;height:100%;border-radius:3px;background:#5be38d;"></div>
      </div>
    </div>
    <div style="flex:1;min-width:80px;">
      <div style="font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">🥑 Fat</div>
      <div style="font-size:0.88rem;font-weight:600;color:#eab308;">{actual_fat_pct}% <span style="color:var(--text-muted);font-weight:400;font-size:0.75rem;">/ target {macro_targets['fat']}%</span></div>
      <div style="background:rgba(26,30,36,0.8);border-radius:3px;height:4px;margin-top:0.3rem;">
        <div style="width:{min(actual_fat_pct/macro_targets['fat']*100,100):.0f}%;height:100%;border-radius:3px;background:#eab308;"></div>
      </div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            # ── Compliance alerts feedback ────────────────────────────────────
            compliance = diet_assistant.get_compliance_feedback(day_nutrition, st.session_state.diet_targets)
            for alert in compliance["alerts"]:
                if alert["type"] == "warning":
                    st.warning(alert["message"])
                elif alert["type"] == "success":
                    st.success(alert["message"])
                else:
                    st.info(alert["message"])

            # ── Budget & Cost Optimization ────────────────────────────────────
            day_costs = budget_optimizer.plan_cost(day_plan)
            daily_cost = day_costs.get("total", 0.0)
            recipe_costs = {k: v for k, v in day_costs.items() if k != "total"}
            budget_feedback = budget_optimizer.get_budget_feedback(daily_cost, recipe_costs, day_plan)

            for alert in budget_feedback["alerts"]:
                if alert["type"] == "warning":
                    st.error(alert["message"])
                elif alert["type"] == "success":
                    st.success(alert["message"])
                else:
                    st.info(alert["message"])
            
            if budget_feedback["swaps"]:
                with st.expander("💡 Save Money with Ingredient Swaps"):
                    st.markdown("Try these cheaper alternatives to lower your meal cost:")
                    for swap in budget_feedback["swaps"]:
                        st.markdown(f"- Swap **{swap['original']}** for **{swap['swap']}** in *{swap['slot'].capitalize()}* (Save **${swap['saving']:.2f}**)")

            # ── Quick stat strip ──────────────────────────────────────────────
            cols = st.columns(5)
            labels = ["💸 Est. Cost", "🔥 Total Calories", "💪 Protein", "🌾 Carbs", "🥑 Fat"]
            vals = [
                f"${daily_cost:.2f}",
                f"{int(day_nutrition['calories'])} kcal",
                f"{day_nutrition['protein']:.0f} g",
                f"{day_nutrition['carbs']:.0f} g",
                f"{day_nutrition['fat']:.0f} g",
            ]
            for col, lbl, val in zip(cols, labels, vals):
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # ── Meal Slots with Regenerate ────────────────────────────────────
            slot_icons = {"breakfast": "☀️", "lunch": "🌤️", "dinner": "🌙"}
            for slot in MEAL_SLOTS:
                recipe = day_plan.get(slot)
                icon = slot_icons[slot]

                slot_col, regen_col = st.columns([5, 1])
                with slot_col:
                    st.markdown(
                        f'<div class="slot-label {slot}">{icon} {slot.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                with regen_col:
                    st.markdown("<div style='padding-top:1.1rem'></div>", unsafe_allow_html=True)
                    if st.button("🔄", key=f"regen_{slot}", help=f"Try a different {slot} recipe"):
                        ings = st.session_state.canonical_ingredients
                        exclude = set(st.session_state.used_titles_today)
                        candidates = matcher.match(
                            user_ingredients=ings,
                            meal_type=slot,
                            dietary_filters=dietary_filters,
                            top_n=10,
                            exclude_titles=exclude,
                            indian_focus=st.session_state.indian_focus,
                        )
                        if not candidates.empty:
                            new_recipe = candidates.iloc[0]
                            st.session_state.day_plan_overrides[slot] = new_recipe
                            st.session_state.used_titles_today.add(new_recipe["title"])
                            invalidate_analysis_cache()
                            invalidate_enrichment_cache()
                            st.rerun()
                        else:
                            st.toast(f"No more {slot} options — try removing dietary filters.", icon="⚠️")

                if recipe is None:
                    st.info(f"No {slot} recipe found with your current ingredients and filters.")
                else:
                    enrichment = get_recipe_enrichment(recipe, st.session_state.canonical_ingredients, rec_engine)
                    render_recipe_card(recipe, slot=slot, enrichment=enrichment)

            # ── Shopping List ─────────────────────────────────────────────────
            st.markdown("---")
            all_missing = []
            seen_missing = set()
            for slot in MEAL_SLOTS:
                recipe = day_plan.get(slot)
                if recipe is not None:
                    raw_missing = recipe.get("missing_ingredients", [])
                    if isinstance(raw_missing, str):
                        raw_missing = [m.strip() for m in raw_missing.split(",") if m.strip()]
                    for m in raw_missing:
                        m_clean = m.replace("_", " ").strip()
                        if m_clean and m_clean not in seen_missing:
                            seen_missing.add(m_clean)
                            all_missing.append(m_clean)

            with st.expander(
                f"🛒 Shopping List ({len(all_missing)} items needed)",
                expanded=len(all_missing) > 0,
            ):
                if not all_missing:
                    st.success("🎉 You have everything you need for today's meals!")
                else:
                    st.markdown(
                        '<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:0.8rem;">'
                        "Items you'll need to buy for today's plan:"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    # Render checklist
                    checklist_html = '<div style="columns:2;gap:1.5rem;">'
                    for item in sorted(all_missing):
                        checklist_html += (
                            f'<div style="margin-bottom:0.4rem;font-size:0.88rem;color:#f1f5f9;">'
                            f'<span style="color:#94a3b8;margin-right:0.5rem;">☐</span>{item.capitalize()}'
                            f'</div>'
                        )
                    checklist_html += "</div>"
                    st.markdown(checklist_html, unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
                    dl_col1, dl_col2 = st.columns(2)

                    # Plain text download
                    txt_content = "Shopping List — Smart Meal Planner\n" + "=" * 38 + "\n\n"
                    for item in sorted(all_missing):
                        txt_content += f"[ ] {item.capitalize()}\n"
                    with dl_col1:
                        st.download_button(
                            label="📄 Download .txt",
                            data=txt_content,
                            file_name="shopping_list.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                    # CSV download
                    import io as _io
                    csv_buf = _io.StringIO()
                    csv_buf.write("Item,Category\n")
                    for item in sorted(all_missing):
                        csv_buf.write(f"{item.capitalize()},To Buy\n")
                    with dl_col2:
                        st.download_button(
                            label="📊 Download .csv",
                            data=csv_buf.getvalue(),
                            file_name="shopping_list.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — WEEKLY PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_weekly:
        if not st.session_state.plan_generated or st.session_state.week_plan is None:
            st.markdown(
                '<div class="empty-state"><div class="icon">📅</div><div class="msg">Generate a meal plan from the Ingredients tab first.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            week_plan = st.session_state.week_plan
            week_nutrition = nutrition.aggregate_week(week_plan)
            total = week_nutrition.get("TOTAL", {})
            week_costs = budget_optimizer.weekly_cost(week_plan)
            week_total_cost = week_costs.get("TOTAL", 0.0)

            # Weekly totals strip
            cols = st.columns(5)
            labels = ["💸 Est. Cost", "🔥 Week Calories", "💪 Protein", "🌾 Carbs", "🥑 Fat"]
            vals = [
                f"${week_total_cost:.2f}",
                f"{int(total.get('calories', 0))} kcal",
                f"{total.get('protein', 0):.0f} g",
                f"{total.get('carbs', 0):.0f} g",
                f"{total.get('fat', 0):.0f} g",
            ]
            for col, lbl, val in zip(cols, labels, vals):
                with col:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            render_weekly_bar_chart(week_nutrition)
            st.markdown("---")

            # Grid: 7 days × 3 meals
            # Show 2 columns of days at a time
            day_list = DAYS
            for i in range(0, 7, 2):
                cols = st.columns(2, gap="large")
                for j, col in enumerate(cols):
                    day_idx = i + j
                    if day_idx >= 7:
                        break
                    day = day_list[day_idx]
                    day_plan = week_plan.get(day, {})
                    day_cost = week_costs.get(day, 0.0)
                    with col:
                        st.markdown(
                            f'<div class="day-header" style="display:flex; justify-content:space-between;">'
                            f'<span>📅 {day}</span>'
                            f'<span style="font-size:0.9rem; color:var(--text-secondary);">Cost: <span style="color:var(--accent-primary);">${day_cost:.2f}</span></span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        slot_icons = {"breakfast": "☀️", "lunch": "🌤️", "dinner": "🌙"}
                        for slot in MEAL_SLOTS:
                            recipe = day_plan.get(slot)
                            if recipe is not None:
                                title = recipe.get("title", "—")
                                cals = int(recipe.get("calories", 0))
                                icon = slot_icons[slot]
                                st.markdown(
                                    f"""
<div class="week-cell">
  <div style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.2rem;">{icon} {slot}</div>
  <div class="week-cell-title">{title}</div>
  <div class="week-cell-cal">🔥 {cals} kcal</div>
</div>
""",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f'<div class="week-cell"><span style="color:#64748b;font-size:0.82rem;">No {slot} found</span></div>',
                                    unsafe_allow_html=True,
                                )
                # Space between rows
                if i < 6:
                    st.markdown("")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — NUTRITION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_nutrition:
        if not st.session_state.plan_generated:
            st.markdown(
                '<div class="empty-state"><div class="icon">📊</div><div class="msg">Generate a meal plan to see nutrition analysis.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            day_plan = get_effective_day_plan()
            week_plan = st.session_state.week_plan

            day_nutrition = nutrition.aggregate_day(day_plan)
            week_nutrition = nutrition.aggregate_week(week_plan)
            total = week_nutrition.get("TOTAL", {})

            # ── Compliance alerts feedback ────────────────────────────────────
            compliance = diet_assistant.get_compliance_feedback(day_nutrition, st.session_state.diet_targets)
            for alert in compliance["alerts"]:
                if alert["type"] == "warning":
                    st.warning(alert["message"])
                elif alert["type"] == "success":
                    st.success(alert["message"])
                else:
                    st.info(alert["message"])

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown(
                    '<div class="section-header">☀️ Today\'s Macros</div>',
                    unsafe_allow_html=True,
                )
                render_nutrition_chart(day_nutrition, "Today's Macro Breakdown")

                # Per-meal breakdown
                st.markdown("**Meal-by-meal calories**")
                for slot in MEAL_SLOTS:
                    recipe = day_plan.get(slot)
                    if recipe is not None:
                        n = nutrition.get_recipe_nutrition(recipe)
                        title = recipe.get("title", "—")
                        cals = int(n["calories"])
                        pct = cals / max(day_nutrition["calories"], 1) * 100
                        st.markdown(
                            f"""
<div style="margin-bottom:0.6rem;">
  <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
    <span style="color:#F1F5F9;font-size:0.85rem;">{title}</span>
    <span style="color:#94a3b8;font-size:0.82rem;">{cals} kcal</span>
  </div>
  <div style="background:#1a1e24;border-radius:4px;height:6px;">
    <div style="width:{pct:.0f}%;background:linear-gradient(90deg,#5be38d,#3db872);height:100%;border-radius:4px;"></div>
  </div>
</div>
""",
                            unsafe_allow_html=True,
                        )

            with col2:
                st.markdown(
                    '<div class="section-header">📅 Weekly Overview</div>',
                    unsafe_allow_html=True,
                )
                render_nutrition_chart(total, "Weekly Macro Breakdown")

                # Daily calorie table
                st.markdown("**Daily calorie breakdown**")
                table_data = []
                for day in DAYS:
                    if day in week_nutrition:
                        dn = week_nutrition[day]
                        table_data.append(
                            {
                                "Day": day,
                                "Calories": f"{int(dn['calories'])} kcal",
                                "Protein": f"{dn['protein']:.0f}g",
                                "Carbs": f"{dn['carbs']:.0f}g",
                                "Fat": f"{dn['fat']:.0f}g",
                            }
                        )
                if table_data:
                    st.dataframe(
                        pd.DataFrame(table_data),
                        use_container_width=True,
                        hide_index=True,
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — DIET ASSISTANT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_diet:
        st.markdown(
            '<div class="section-header">🥗 Smart Diet & Nutrition Assistant</div>',
            unsafe_allow_html=True,
        )
        col_profile, col_coach = st.columns([1, 1], gap="large")

        with col_profile:
            st.markdown("### 📊 Your Health Profile")
            st.markdown('<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;margin-bottom:1.5rem;">', unsafe_allow_html=True)
            
            p_gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.diet_profile["gender"] == "Male" else 1)
            p_age = st.number_input("Age (years)", min_value=1, max_value=120, value=st.session_state.diet_profile["age"])
            p_weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=st.session_state.diet_profile["weight"], step=0.1)
            p_height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=st.session_state.diet_profile["height"], step=0.5)
            p_activity = st.selectbox("Activity Level", list(DietAssistantModule.ACTIVITY_MULTIPLIERS.keys()), index=list(DietAssistantModule.ACTIVITY_MULTIPLIERS.keys()).index(st.session_state.diet_profile["activity"]))
            p_goal = st.selectbox("Diet Goal", list(DietAssistantModule.GOAL_MULTIPLIERS.keys()), index=list(DietAssistantModule.GOAL_MULTIPLIERS.keys()).index(st.session_state.diet_profile["goal"]))
            p_macro = st.selectbox("Macronutrient Preset Target", list(DietAssistantModule.MACRO_PRESETS.keys()), index=list(DietAssistantModule.MACRO_PRESETS.keys()).index(st.session_state.diet_profile["macro_preset"]))

            if st.button("🎯 Calculate & Apply Targets", use_container_width=True):
                # Calculate targets
                calculated = diet_assistant.calculate_targets(
                    weight_kg=p_weight,
                    height_cm=p_height,
                    age_years=p_age,
                    gender=p_gender,
                    activity_level=p_activity,
                    diet_goal=p_goal,
                    macro_preset=p_macro
                )
                # Store back
                st.session_state.diet_profile = {
                    "age": p_age,
                    "gender": p_gender,
                    "weight": p_weight,
                    "height": p_height,
                    "activity": p_activity,
                    "goal": p_goal,
                    "macro_preset": p_macro
                }
                st.session_state.diet_targets = calculated
                # Sync back to global states
                st.session_state.calorie_goal = int(calculated["target_calories"])
                st.session_state.macro_targets = calculated["macro_percentages"]
                st.success("✅ Targets calculated and applied globally!")
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

            # Display calculated target details
            t_data = st.session_state.diet_targets
            st.markdown(
                f"""
<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;box-shadow:var(--shadow-card);">
  <div style="font-size:0.95rem;font-weight:700;color:#5be38d;margin-bottom:0.8rem;">Calculated Daily Targets</div>
  <div style="display:grid;grid-template-columns: 1fr 1fr; gap:0.8rem; margin-bottom:1rem;">
    <div style="background:rgba(255,255,255,0.02);padding:0.6rem;border-radius:6px;border:1px solid var(--border);">
      <span style="font-size:0.68rem;color:var(--text-muted);display:block;text-transform:uppercase;">Basal Metabolic Rate</span>
      <span style="font-size:1.15rem;font-weight:700;color:var(--text-primary);">{t_data['bmr']:.1f} <span style="font-size:0.75rem;font-weight:normal;color:var(--text-muted);">kcal</span></span>
    </div>
    <div style="background:rgba(255,255,255,0.02);padding:0.6rem;border-radius:6px;border:1px solid var(--border);">
      <span style="font-size:0.68rem;color:var(--text-muted);display:block;text-transform:uppercase;">Daily Maintenance (TDEE)</span>
      <span style="font-size:1.15rem;font-weight:700;color:var(--text-primary);">{t_data['tdee']:.1f} <span style="font-size:0.75rem;font-weight:normal;color:var(--text-muted);">kcal</span></span>
    </div>
  </div>
  <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.2rem;text-transform:uppercase;">Personal Target Limit</div>
  <div style="font-size:1.8rem;font-weight:800;color:#5be38d;margin-bottom:1rem;">{int(t_data['target_calories'])} <span style="font-size:0.9rem;font-weight:normal;color:var(--text-muted);">kcal / day</span></div>
  <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.4rem;text-transform:uppercase;">Target Macronutrient Targets</div>
  <div style="display:flex;gap:0.5rem;">
    <div style="flex:1;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.15);padding:0.5rem;border-radius:8px;text-align:center;">
      <span style="font-size:0.65rem;color:#3b82f6;display:block;text-transform:uppercase;font-weight:bold;">Carbs ({t_data['macro_percentages']['carbs']}%)</span>
      <span style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">{t_data['macro_targets']['carbs']}g</span>
    </div>
    <div style="flex:1;background:rgba(91,227,141,0.06);border:1px solid rgba(91,227,141,0.15);padding:0.5rem;border-radius:8px;text-align:center;">
      <span style="font-size:0.65rem;color:#5be38d;display:block;text-transform:uppercase;font-weight:bold;">Protein ({t_data['macro_percentages']['protein']}%)</span>
      <span style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">{t_data['macro_targets']['protein']}g</span>
    </div>
    <div style="flex:1;background:rgba(234,179,8,0.06);border:1px solid rgba(234,179,8,0.15);padding:0.5rem;border-radius:8px;text-align:center;">
      <span style="font-size:0.65rem;color:#eab308;display:block;text-transform:uppercase;font-weight:bold;">Fat ({t_data['macro_percentages']['fat']}%)</span>
      <span style="font-size:1.1rem;font-weight:700;color:#F1F5F9;">{t_data['macro_targets']['fat']}g</span>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        with col_coach:
            st.markdown("### 💬 Diet & Nutrition Coach")
            st.info("💡 Get advice on meeting your weight goals, ask about recipe replacements, or review daily plan macro deficits.")

            # Render diet chat history
            for msg in st.session_state.diet_chat_messages:
                with st.chat_message(msg["role"], avatar="🥦" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask your Diet Coach anything...", key="diet_chat_input"):
                st.session_state.diet_chat_messages.append({"role": "user", "content": prompt})

                with custom_spinner("Dietitian is analyzing..."):
                    day_plan = get_effective_day_plan()
                    summary = "No meal plan generated yet."
                    if day_plan:
                        summary_list = []
                        for slot in MEAL_SLOTS:
                            recipe = day_plan.get(slot)
                            if recipe is not None:
                                n = nutrition.get_recipe_nutrition(recipe)
                                summary_list.append(f"- {slot.upper()}: {recipe.get('title')} ({int(n['calories'])} kcal, {n['protein']}g P, {n['carbs']}g C, {n['fat']}g F)")
                        summary = "\n".join(summary_list)
                    
                    sys_prompt = diet_assistant.generate_dietitian_system_prompt(
                        user_profile=st.session_state.diet_profile,
                        targets=st.session_state.diet_targets,
                        daily_plan_summary=summary
                    )

                    # Use custom dietitian instruction inside chat_with_chef
                    coach_response = intent_parser.chat_with_chef(
                        messages=st.session_state.diet_chat_messages,
                        system_msg=sys_prompt
                    )

                st.session_state.diet_chat_messages.append({"role": "assistant", "content": coach_response})
                st.write("<div style='height: 100px'></div>", unsafe_allow_html=True)
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — AI CHEF
    # ══════════════════════════════════════════════════════════════════════════
    with tab_chat:
        st.markdown(
            '<div class="section-header">💬 Chat with your AI Chef</div>',
            unsafe_allow_html=True,
        )
        st.info("💡 You can ask for meal ideas, substitutions, or tell me what ingredients you have!")
        
        # Render chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"], avatar="👨‍🍳" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])
                
        # Chat input
        if prompt := st.chat_input("Ask your chef anything..."):
            # Add user message to state
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            # Temporary placeholder for thinking
            with custom_spinner("Chef is thinking..."):
                quick = _quick_chat_intent(prompt)

                if quick == "CHAT":
                    final_response = intent_parser.chat_with_chef(st.session_state.chat_messages)
                else:
                    parsed = intent_parser.parse_intent(prompt)
                    intent = parsed.get("intent", "CHAT")
                    extracted_ing = _extract_ingredients_from_text(prompt, parsed, normalizer)
                    final_response = ""

                    if extracted_ing:
                        canonical = normalizer.get_ingredient_set(", ".join(extracted_ing))
                        if canonical:
                            current_ings = set(st.session_state.canonical_ingredients or [])
                            st.session_state.canonical_ingredients = list(current_ings.union(canonical))
                            update_ingredient_quantities(extracted_ing)
                            invalidate_analysis_cache()

                    if intent == "ADD_INGREDIENTS" or (quick == "ACTION" and extracted_ing):
                        if st.session_state.canonical_ingredients:
                            final_response = (
                                f"I've noted these ingredients. You now have "
                                f"{len(st.session_state.canonical_ingredients)} items in your pantry. "
                                "Shall I generate a meal plan?"
                            )
                        else:
                            final_response = "I didn't catch any specific ingredients. What do you have?"

                    elif intent == "GENERATE_MEAL" or quick == "GENERATE_MEAL":
                        if st.session_state.canonical_ingredients:
                            _apply_meal_plan(
                                st.session_state.canonical_ingredients,
                                planner,
                                dietary_filters,
                                st.session_state.indian_focus,
                            )
                            final_response = (
                                "Your meal plan is ready! Switch to the 'Today's Plan' "
                                "or 'Weekly Plan' tab to see it."
                            )
                        else:
                            final_response = (
                                "Please tell me what ingredients you have before I can generate a meal for you!"
                            )
                    elif parsed.get("chat_response"):
                        final_response = parsed["chat_response"]
                    else:
                        final_response = intent_parser.chat_with_chef(st.session_state.chat_messages)
                    
            st.session_state.chat_messages.append({"role": "assistant", "content": final_response})
            
            # Spacer so last message isn't hidden behind sticky input
            st.write("<div style='height: 100px'></div>", unsafe_allow_html=True)
            st.rerun()



if __name__ == "__main__":
    main()

# trigger reload for chip CSS

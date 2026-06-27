"""
Recipe Generator — produces 100,000+ realistic recipes.
Run once to generate data/recipes.csv
"""

import csv
import random
import os
import sys
from itertools import product

random.seed(42)

# ── Ingredient Database ─────────────────────────────────────────────────────

PROTEINS = [
    "chicken_breast", "chicken_thigh", "ground_chicken", "ground_beef",
    "beef_strips", "lamb", "ground_lamb", "pork", "ground_pork",
    "shrimp", "salmon", "tuna", "cod", "tilapia", "prawns",
    "egg", "tofu", "paneer", "turkey", "duck",
    "sausage", "bacon", "ham", "crab", "lobster",
]

VEGETABLES = [
    "tomato", "onion", "garlic", "bell_pepper", "carrot", "broccoli",
    "spinach", "potato", "mushroom", "zucchini", "cucumber", "corn",
    "peas", "eggplant", "cauliflower", "cabbage", "lettuce",
    "celery", "asparagus", "green_bean", "kale", "sweet_potato",
    "radish", "beetroot", "pumpkin", "okra", "spring_onion",
    "ginger", "chili", "jalapeno", "avocado", "artichoke",
    "leek", "turnip", "parsnip", "brussels_sprout", "bok_choy",
    "bean_sprout", "water_chestnut", "bamboo_shoot",
]

GRAINS = [
    "rice", "pasta", "bread", "flour", "oats", "quinoa", "tortilla",
    "noodles", "couscous", "barley", "cornmeal", "pizza_dough",
    "pita_bread", "roti", "bulgur", "semolina", "breadcrumbs",
    "rice_noodles", "udon_noodles", "soba_noodles",
]

DAIRY = [
    "milk", "butter", "cheese", "cream", "yogurt", "sour_cream",
    "mozzarella", "parmesan", "cheddar_cheese", "cream_cheese",
    "feta_cheese", "ricotta", "ghee", "coconut_milk", "whipping_cream",
    "greek_yogurt", "cottage_cheese", "gouda",
]

HERBS_SPICES = [
    "salt", "pepper", "cumin", "turmeric", "garam_masala", "oregano",
    "basil", "thyme", "rosemary", "cinnamon", "chili_powder", "paprika",
    "coriander", "cardamom", "clove", "nutmeg", "bay_leaf",
    "curry_powder", "cayenne", "dill", "parsley", "mint",
    "sage", "tarragon", "fennel_seeds", "mustard_seeds",
    "star_anise", "saffron", "sumac", "za_atar",
    "italian_seasoning", "five_spice", "lemongrass",
]

SAUCES_CONDIMENTS = [
    "olive_oil", "sesame_oil", "vegetable_oil", "coconut_oil",
    "soy_sauce", "vinegar", "balsamic_vinegar", "lemon", "lime",
    "honey", "maple_syrup", "mustard", "ketchup", "mayonnaise",
    "hot_sauce", "sriracha", "fish_sauce", "oyster_sauce",
    "tomato_sauce", "tomato_paste", "salsa", "tahini",
    "worcestershire_sauce", "miso_paste", "teriyaki_sauce",
    "peanut_butter", "hoisin_sauce", "chili_paste",
    "ranch_dressing", "caesar_dressing", "barbecue_sauce",
]

LEGUMES = [
    "lentil", "chickpea", "black_bean", "kidney_bean", "white_bean",
    "pinto_bean", "edamame", "split_pea", "black_eyed_pea",
    "mung_bean", "red_lentil", "green_lentil",
]

NUTS_SEEDS = [
    "almond", "cashew", "peanut", "walnut", "pistachio",
    "chia_seeds", "sesame_seeds", "sunflower_seeds", "flax_seeds",
    "pine_nuts", "coconut_flakes", "pumpkin_seeds", "pecan",
]

FRUITS = [
    "banana", "mango", "apple", "strawberry", "blueberry",
    "orange", "pineapple", "coconut", "pomegranate", "grape",
    "peach", "pear", "watermelon", "papaya", "kiwi",
    "raspberry", "cranberry", "fig", "date", "raisin",
]

# ── Cuisine Profiles ────────────────────────────────────────────────────────

CUISINES = {
    "indian": {
        "adj": ["Indian", "Masala", "Tandoori", "Curry", "Spiced", "Desi"],
        "proteins": ["chicken_breast", "chicken_thigh", "lamb", "ground_lamb", "egg", "paneer", "shrimp", "prawns"],
        "vegetables": ["tomato", "onion", "garlic", "potato", "spinach", "cauliflower", "peas", "eggplant", "okra", "bell_pepper", "ginger", "chili", "cabbage", "mushroom", "carrot", "corn", "beetroot", "pumpkin", "sweet_potato", "radish"],
        "grains": ["rice", "roti", "flour", "bread", "semolina"],
        "dairy": ["butter", "ghee", "yogurt", "cream", "milk", "paneer", "coconut_milk", "greek_yogurt"],
        "spices": ["salt", "pepper", "cumin", "turmeric", "garam_masala", "coriander", "chili_powder", "cardamom", "mustard_seeds", "curry_powder", "cayenne", "fennel_seeds"],
        "sauces": ["olive_oil", "vegetable_oil", "lemon", "tomato_paste", "tomato_sauce"],
        "legumes": ["lentil", "chickpea", "kidney_bean", "mung_bean", "red_lentil", "green_lentil", "black_eyed_pea", "split_pea"],
        "tags": ["gluten-free"],
        "methods": ["Spiced", "Curried", "Tandoori", "Masala", "Roasted", "Fried", "Sautéed", "Braised", "Steamed", "Grilled"],
        "dishes": ["Curry", "Masala", "Fry", "Sabzi", "Rice Bowl", "Wrap", "Stew", "Biryani", "Tikka", "Korma", "Bhurji", "Dal", "Pulao"],
    },
    "italian": {
        "adj": ["Italian", "Tuscan", "Roman", "Sicilian", "Rustic"],
        "proteins": ["chicken_breast", "ground_beef", "shrimp", "salmon", "sausage", "egg", "turkey"],
        "vegetables": ["tomato", "onion", "garlic", "bell_pepper", "mushroom", "zucchini", "spinach", "eggplant", "artichoke", "asparagus", "broccoli", "kale", "celery"],
        "grains": ["pasta", "bread", "pizza_dough", "flour", "breadcrumbs", "rice"],
        "dairy": ["mozzarella", "parmesan", "ricotta", "cream", "butter", "cream_cheese"],
        "spices": ["salt", "pepper", "oregano", "basil", "italian_seasoning", "thyme", "rosemary", "bay_leaf", "parsley"],
        "sauces": ["olive_oil", "tomato_sauce", "tomato_paste", "balsamic_vinegar", "lemon", "vinegar"],
        "legumes": ["white_bean", "chickpea", "kidney_bean"],
        "tags": [],
        "methods": ["Baked", "Grilled", "Sautéed", "Roasted", "Pan-Fried", "Braised", "Tossed"],
        "dishes": ["Pasta", "Risotto", "Soup", "Salad", "Bake", "Frittata", "Bruschetta", "Penne", "Spaghetti", "Linguine"],
    },
    "mexican": {
        "adj": ["Mexican", "Tex-Mex", "Baja", "Chipotle", "Spicy"],
        "proteins": ["chicken_breast", "ground_beef", "shrimp", "pork", "egg", "turkey", "ground_pork"],
        "vegetables": ["tomato", "onion", "garlic", "bell_pepper", "corn", "jalapeno", "avocado", "lettuce", "cabbage", "chili", "sweet_potato", "potato", "zucchini"],
        "grains": ["tortilla", "rice", "cornmeal", "bread", "flour"],
        "dairy": ["cheese", "sour_cream", "cream", "cheddar_cheese", "cream_cheese"],
        "spices": ["salt", "pepper", "cumin", "chili_powder", "paprika", "cayenne", "coriander", "oregano"],
        "sauces": ["salsa", "lime", "olive_oil", "hot_sauce", "tomato_sauce", "vegetable_oil"],
        "legumes": ["black_bean", "pinto_bean", "kidney_bean"],
        "tags": [],
        "methods": ["Grilled", "Fried", "Baked", "Roasted", "Spiced", "Charred", "Stuffed"],
        "dishes": ["Tacos", "Burrito", "Quesadilla", "Bowl", "Enchilada", "Fajitas", "Nachos", "Wrap", "Salad"],
    },
    "chinese": {
        "adj": ["Chinese", "Szechuan", "Cantonese", "Hunan", "Wok"],
        "proteins": ["chicken_breast", "beef_strips", "shrimp", "pork", "tofu", "egg", "duck", "prawns"],
        "vegetables": ["garlic", "ginger", "onion", "bell_pepper", "broccoli", "mushroom", "bok_choy", "bean_sprout", "water_chestnut", "bamboo_shoot", "cabbage", "carrot", "corn", "spring_onion", "zucchini", "eggplant", "spinach", "celery", "chili"],
        "grains": ["rice", "noodles", "rice_noodles", "flour"],
        "dairy": [],
        "spices": ["salt", "pepper", "five_spice", "star_anise", "ginger", "chili_powder"],
        "sauces": ["soy_sauce", "sesame_oil", "vegetable_oil", "oyster_sauce", "hoisin_sauce", "chili_paste", "vinegar", "honey", "sriracha"],
        "legumes": ["edamame", "mung_bean"],
        "tags": [],
        "methods": ["Stir-Fried", "Steamed", "Braised", "Fried", "Wok-Tossed", "Glazed", "Crispy"],
        "dishes": ["Stir-Fry", "Fried Rice", "Noodles", "Soup", "Bowl", "Dumplings", "Lo Mein", "Chow Mein"],
    },
    "japanese": {
        "adj": ["Japanese", "Tokyo", "Kyoto", "Teriyaki", "Miso"],
        "proteins": ["salmon", "tuna", "shrimp", "chicken_breast", "tofu", "egg", "cod", "prawns"],
        "vegetables": ["garlic", "ginger", "onion", "mushroom", "spinach", "cucumber", "carrot", "cabbage", "spring_onion", "bean_sprout", "bok_choy", "radish", "eggplant", "sweet_potato", "avocado", "corn", "edamame"],
        "grains": ["rice", "udon_noodles", "soba_noodles", "noodles", "breadcrumbs"],
        "dairy": [],
        "spices": ["salt", "pepper", "ginger"],
        "sauces": ["soy_sauce", "sesame_oil", "miso_paste", "teriyaki_sauce", "vinegar", "honey", "lime", "sriracha", "vegetable_oil"],
        "legumes": ["edamame"],
        "tags": [],
        "methods": ["Teriyaki", "Grilled", "Pan-Fried", "Steamed", "Glazed", "Seared", "Tempura"],
        "dishes": ["Bowl", "Ramen", "Stir-Fry", "Salad", "Soup", "Rice Bowl", "Udon", "Soba"],
    },
    "thai": {
        "adj": ["Thai", "Bangkok", "Spicy Thai", "Sweet Thai"],
        "proteins": ["chicken_breast", "shrimp", "tofu", "beef_strips", "pork", "prawns", "egg"],
        "vegetables": ["garlic", "ginger", "onion", "bell_pepper", "mushroom", "bean_sprout", "bok_choy", "cabbage", "carrot", "corn", "spinach", "bamboo_shoot", "spring_onion", "chili", "eggplant", "zucchini", "sweet_potato", "avocado"],
        "grains": ["rice", "rice_noodles", "noodles"],
        "dairy": ["coconut_milk"],
        "spices": ["salt", "pepper", "lemongrass", "chili_powder", "coriander", "cumin", "turmeric", "curry_powder"],
        "sauces": ["soy_sauce", "fish_sauce", "sesame_oil", "lime", "honey", "peanut_butter", "sriracha", "chili_paste", "vegetable_oil", "coconut_oil"],
        "legumes": ["edamame", "mung_bean"],
        "tags": ["gluten-free"],
        "methods": ["Thai", "Stir-Fried", "Curried", "Grilled", "Steamed", "Spiced", "Sautéed"],
        "dishes": ["Curry", "Stir-Fry", "Noodles", "Soup", "Pad Thai", "Bowl", "Salad", "Fried Rice"],
    },
    "american": {
        "adj": ["American", "Classic", "Homestyle", "Southern", "BBQ"],
        "proteins": ["chicken_breast", "ground_beef", "turkey", "bacon", "ham", "sausage", "egg", "salmon", "shrimp", "pork", "ground_pork"],
        "vegetables": ["tomato", "onion", "garlic", "potato", "corn", "lettuce", "mushroom", "bell_pepper", "celery", "carrot", "cabbage", "broccoli", "spinach", "sweet_potato", "green_bean", "asparagus", "zucchini", "cucumber", "avocado", "peas"],
        "grains": ["bread", "flour", "rice", "pasta", "oats", "cornmeal", "tortilla", "pizza_dough"],
        "dairy": ["butter", "cheese", "milk", "cream", "cheddar_cheese", "sour_cream", "cream_cheese", "mozzarella"],
        "spices": ["salt", "pepper", "paprika", "oregano", "thyme", "rosemary", "chili_powder", "garlic", "parsley", "bay_leaf", "dill", "cayenne"],
        "sauces": ["olive_oil", "vegetable_oil", "ketchup", "mustard", "mayonnaise", "barbecue_sauce", "hot_sauce", "worcestershire_sauce", "lemon", "vinegar", "honey", "ranch_dressing", "tomato_sauce"],
        "legumes": ["black_bean", "kidney_bean", "white_bean"],
        "tags": [],
        "methods": ["Grilled", "Baked", "Fried", "Roasted", "Pan-Seared", "BBQ", "Smoked", "Sautéed", "Braised", "Steamed"],
        "dishes": ["Steak", "Burger", "Soup", "Salad", "Sandwich", "Bowl", "Stew", "Casserole", "Wrap", "Bake", "Chowder", "Hash"],
    },
    "french": {
        "adj": ["French", "Provençal", "Parisian", "Bistro", "Rustic French"],
        "proteins": ["chicken_breast", "duck", "salmon", "shrimp", "egg", "lamb", "beef_strips", "cod"],
        "vegetables": ["onion", "garlic", "mushroom", "leek", "asparagus", "spinach", "potato", "carrot", "celery", "tomato", "zucchini", "bell_pepper", "artichoke", "green_bean", "turnip", "parsnip"],
        "grains": ["bread", "flour", "pasta"],
        "dairy": ["butter", "cream", "cheese", "cream_cheese", "gouda", "parmesan", "milk"],
        "spices": ["salt", "pepper", "thyme", "rosemary", "tarragon", "bay_leaf", "parsley", "dill", "nutmeg", "sage"],
        "sauces": ["olive_oil", "lemon", "vinegar", "mustard", "balsamic_vinegar", "honey"],
        "legumes": ["white_bean", "lentil", "green_lentil"],
        "tags": [],
        "methods": ["Sautéed", "Braised", "Roasted", "Pan-Seared", "Poached", "Baked", "Grilled", "Flambéed"],
        "dishes": ["Soup", "Salad", "Gratin", "Ratatouille", "Stew", "Quiche", "Tart", "Bisque", "Fricassée"],
    },
    "mediterranean": {
        "adj": ["Mediterranean", "Greek", "Aegean", "Levantine", "Coastal"],
        "proteins": ["chicken_breast", "lamb", "shrimp", "salmon", "egg", "cod", "ground_lamb"],
        "vegetables": ["tomato", "onion", "garlic", "cucumber", "bell_pepper", "eggplant", "spinach", "zucchini", "artichoke", "potato", "lettuce", "mushroom", "radish", "beetroot", "asparagus", "avocado", "celery"],
        "grains": ["pita_bread", "rice", "couscous", "bread", "bulgur", "quinoa", "pasta"],
        "dairy": ["feta_cheese", "yogurt", "cream", "butter", "mozzarella", "greek_yogurt"],
        "spices": ["salt", "pepper", "oregano", "cumin", "paprika", "sumac", "za_atar", "parsley", "mint", "dill", "thyme", "coriander", "rosemary"],
        "sauces": ["olive_oil", "lemon", "tahini", "vinegar", "balsamic_vinegar", "honey"],
        "legumes": ["chickpea", "lentil", "white_bean", "green_lentil"],
        "tags": [],
        "methods": ["Grilled", "Roasted", "Baked", "Sautéed", "Marinated", "Stuffed", "Braised"],
        "dishes": ["Salad", "Bowl", "Wrap", "Soup", "Bake", "Platter", "Stew", "Mezze"],
    },
    "korean": {
        "adj": ["Korean", "Seoul", "Spicy Korean", "K-BBQ"],
        "proteins": ["chicken_breast", "beef_strips", "pork", "tofu", "egg", "shrimp", "ground_beef", "ground_pork"],
        "vegetables": ["garlic", "ginger", "onion", "spring_onion", "cabbage", "carrot", "spinach", "mushroom", "cucumber", "bell_pepper", "bean_sprout", "radish", "zucchini", "corn", "chili", "sweet_potato", "lettuce"],
        "grains": ["rice", "noodles", "flour", "rice_noodles"],
        "dairy": [],
        "spices": ["salt", "pepper", "chili_powder", "ginger", "sesame_seeds"],
        "sauces": ["soy_sauce", "sesame_oil", "vegetable_oil", "vinegar", "honey", "sriracha", "chili_paste", "hot_sauce"],
        "legumes": ["edamame", "mung_bean"],
        "tags": [],
        "methods": ["Korean BBQ", "Stir-Fried", "Braised", "Grilled", "Steamed", "Spiced", "Crispy"],
        "dishes": ["Bibimbap", "Bowl", "Stir-Fry", "Fried Rice", "Soup", "Noodles", "Wrap", "Salad"],
    },
    "middle_eastern": {
        "adj": ["Middle Eastern", "Lebanese", "Turkish", "Persian"],
        "proteins": ["chicken_breast", "lamb", "ground_lamb", "ground_beef", "egg", "shrimp"],
        "vegetables": ["tomato", "onion", "garlic", "cucumber", "eggplant", "bell_pepper", "spinach", "potato", "carrot", "cabbage", "lettuce", "radish", "beetroot", "mushroom", "cauliflower", "okra", "zucchini"],
        "grains": ["rice", "pita_bread", "bulgur", "couscous", "flour", "bread"],
        "dairy": ["yogurt", "feta_cheese", "cream", "butter", "greek_yogurt"],
        "spices": ["salt", "pepper", "cumin", "coriander", "paprika", "sumac", "za_atar", "turmeric", "cardamom", "cinnamon", "mint", "parsley"],
        "sauces": ["olive_oil", "lemon", "tahini", "vinegar", "tomato_paste", "honey"],
        "legumes": ["chickpea", "lentil", "red_lentil", "white_bean"],
        "tags": [],
        "methods": ["Grilled", "Roasted", "Spiced", "Baked", "Stuffed", "Braised", "Sautéed"],
        "dishes": ["Kebab", "Bowl", "Wrap", "Salad", "Soup", "Stew", "Platter", "Hummus", "Falafel"],
    },
    "spanish": {
        "adj": ["Spanish", "Catalan", "Andalusian"],
        "proteins": ["chicken_breast", "shrimp", "chorizo_sausage", "egg", "cod", "salmon", "pork", "lamb", "sausage"],
        "vegetables": ["tomato", "onion", "garlic", "bell_pepper", "potato", "spinach", "mushroom", "artichoke", "asparagus", "zucchini", "eggplant", "peas", "corn", "lettuce"],
        "grains": ["rice", "bread", "flour", "pasta"],
        "dairy": ["cheese", "cream", "butter", "mozzarella"],
        "spices": ["salt", "pepper", "paprika", "saffron", "oregano", "bay_leaf", "parsley", "cumin", "thyme", "rosemary"],
        "sauces": ["olive_oil", "lemon", "vinegar", "tomato_sauce", "tomato_paste"],
        "legumes": ["chickpea", "white_bean", "lentil"],
        "tags": [],
        "methods": ["Grilled", "Baked", "Sautéed", "Roasted", "Pan-Fried", "Braised", "Seared"],
        "dishes": ["Paella", "Tapas", "Soup", "Salad", "Stew", "Tortilla", "Bake", "Bowl"],
    },
    "vietnamese": {
        "adj": ["Vietnamese", "Saigon", "Hanoi"],
        "proteins": ["chicken_breast", "beef_strips", "shrimp", "pork", "tofu", "egg", "prawns"],
        "vegetables": ["garlic", "ginger", "onion", "bean_sprout", "carrot", "cucumber", "lettuce", "cabbage", "mushroom", "spring_onion", "bell_pepper", "chili", "corn", "bok_choy", "radish", "spinach"],
        "grains": ["rice", "rice_noodles", "noodles", "bread"],
        "dairy": [],
        "spices": ["salt", "pepper", "lemongrass", "coriander", "chili_powder", "mint", "basil"],
        "sauces": ["soy_sauce", "fish_sauce", "lime", "sesame_oil", "sriracha", "hoisin_sauce", "vegetable_oil", "peanut_butter", "vinegar", "honey"],
        "legumes": ["mung_bean", "edamame"],
        "tags": ["gluten-free"],
        "methods": ["Grilled", "Stir-Fried", "Steamed", "Fresh", "Braised", "Pan-Fried", "Sautéed"],
        "dishes": ["Pho", "Bowl", "Spring Roll", "Stir-Fry", "Noodles", "Salad", "Soup", "Banh Mi", "Fried Rice"],
    },
    "british": {
        "adj": ["British", "English", "Classic British"],
        "proteins": ["chicken_breast", "ground_beef", "lamb", "cod", "salmon", "sausage", "bacon", "ham", "egg", "turkey"],
        "vegetables": ["potato", "onion", "garlic", "carrot", "peas", "mushroom", "cabbage", "broccoli", "celery", "leek", "turnip", "parsnip", "brussels_sprout", "cauliflower", "tomato", "corn", "spinach", "green_bean", "asparagus"],
        "grains": ["bread", "flour", "oats", "pasta", "rice"],
        "dairy": ["butter", "milk", "cheese", "cream", "cheddar_cheese"],
        "spices": ["salt", "pepper", "thyme", "rosemary", "sage", "bay_leaf", "parsley", "mustard_seeds", "nutmeg", "paprika", "dill"],
        "sauces": ["vegetable_oil", "vinegar", "worcestershire_sauce", "mustard", "ketchup", "lemon", "olive_oil", "honey"],
        "legumes": ["kidney_bean", "white_bean", "lentil", "split_pea"],
        "tags": [],
        "methods": ["Roasted", "Baked", "Grilled", "Fried", "Braised", "Steamed", "Pan-Fried", "Sautéed", "Boiled"],
        "dishes": ["Pie", "Soup", "Stew", "Roast", "Bake", "Hash", "Salad", "Sandwich", "Casserole", "Chowder"],
    },
    "caribbean": {
        "adj": ["Caribbean", "Jamaican", "Island", "Tropical"],
        "proteins": ["chicken_breast", "chicken_thigh", "shrimp", "pork", "cod", "prawns", "egg", "ground_beef"],
        "vegetables": ["tomato", "onion", "garlic", "bell_pepper", "sweet_potato", "corn", "cabbage", "spinach", "avocado", "chili", "ginger", "spring_onion", "carrot", "pumpkin", "mushroom", "cucumber", "lettuce", "potato"],
        "grains": ["rice", "bread", "flour", "tortilla"],
        "dairy": ["butter", "coconut_milk", "cream"],
        "spices": ["salt", "pepper", "cumin", "paprika", "thyme", "cinnamon", "nutmeg", "cayenne", "coriander", "chili_powder"],
        "sauces": ["lime", "lemon", "vegetable_oil", "coconut_oil", "honey", "hot_sauce", "vinegar", "soy_sauce"],
        "legumes": ["black_bean", "kidney_bean", "pinto_bean", "lentil", "chickpea"],
        "tags": ["gluten-free"],
        "methods": ["Jerk", "Grilled", "Fried", "Braised", "Roasted", "Curried", "Spiced", "Steamed"],
        "dishes": ["Curry", "Rice Bowl", "Stew", "Wrap", "Salad", "Soup", "Bowl", "Fry"],
    },
}

MEAL_TYPE_RULES = {
    "breakfast": {
        "preferred_proteins": ["egg", "bacon", "sausage", "ham", "turkey", "paneer", "tofu"],
        "preferred_grains": ["bread", "flour", "oats", "tortilla", "roti", "semolina", "pita_bread"],
        "preferred_dairy": ["butter", "milk", "cheese", "yogurt", "cream", "ghee", "cream_cheese", "greek_yogurt"],
        "max_ingredients": 7,
        "calorie_range": (150, 450),
    },
    "lunch": {
        "preferred_proteins": None,
        "preferred_grains": None,
        "preferred_dairy": None,
        "max_ingredients": 9,
        "calorie_range": (200, 600),
    },
    "dinner": {
        "preferred_proteins": None,
        "preferred_grains": None,
        "preferred_dairy": None,
        "max_ingredients": 10,
        "calorie_range": (250, 700),
    },
}

VEGETARIAN_PROTEINS = {"egg", "tofu", "paneer"}
VEGAN_EXCLUDE = {"egg", "tofu", "paneer", "butter", "milk", "cheese", "cream", "yogurt",
                 "sour_cream", "mozzarella", "parmesan", "cheddar_cheese", "cream_cheese",
                 "feta_cheese", "ricotta", "ghee", "whipping_cream", "greek_yogurt",
                 "cottage_cheese", "gouda", "honey", "mayonnaise", "fish_sauce"}

ALL_MEATS = set(PROTEINS) - VEGETARIAN_PROTEINS


def generate_tags(ingredients, cuisine_tags):
    """Generate dietary tags based on ingredients."""
    ing_set = set(ingredients)
    tags = list(cuisine_tags)

    has_meat = bool(ing_set & ALL_MEATS)
    has_non_vegan = bool(ing_set & VEGAN_EXCLUDE)

    if not has_meat:
        tags.append("vegetarian")
        if not has_non_vegan:
            tags.append("vegan")

    if not ing_set & {"flour", "bread", "pasta", "tortilla", "noodles", "couscous",
                      "barley", "pizza_dough", "pita_bread", "breadcrumbs", "bulgur",
                      "semolina", "roti", "udon_noodles", "soba_noodles", "oats", "cornmeal"}:
        tags.append("gluten-free")

    if len(ingredients) <= 5:
        tags.append("quick")

    return ",".join(sorted(set(tags)))


def estimate_nutrition(ingredients, meal_type):
    """Simple calorie/macro estimation."""
    cal_min, cal_max = MEAL_TYPE_RULES[meal_type]["calorie_range"]
    n = len(ingredients)

    base_cal = random.randint(cal_min, cal_max)
    has_protein = bool(set(ingredients) & set(PROTEINS))
    has_grain = bool(set(ingredients) & set(GRAINS))

    protein = random.randint(8, 20)
    carbs = random.randint(10, 40)
    fat = random.randint(4, 18)

    if has_protein:
        protein = random.randint(20, 45)
        base_cal += random.randint(40, 100)
    if has_grain:
        carbs = random.randint(40, 75)
        base_cal += random.randint(30, 80)

    calories = max(cal_min, min(cal_max + 100, base_cal))
    return calories, protein, carbs, fat


def generate_instruction(method, ingredients, dish_type):
    """Generate a short cooking instruction."""
    protein_ings = [i for i in ingredients if i in set(PROTEINS)]
    veg_ings = [i for i in ingredients if i in set(VEGETABLES)]
    grain_ings = [i for i in ingredients if i in set(GRAINS)]

    p_str = protein_ings[0].replace("_", " ") if protein_ings else ""
    v_str = " and ".join(i.replace("_", " ") for i in veg_ings[:2]) if veg_ings else ""
    g_str = grain_ings[0].replace("_", " ") if grain_ings else ""

    templates = [
        f"{method} the {p_str}, combine with {v_str}, season and serve." if p_str and v_str else None,
        f"Prepare {v_str}, cook until tender, serve as {dish_type.lower()}." if v_str else None,
        f"Cook {g_str}, add {p_str or v_str}, season well." if g_str else None,
        f"Season {p_str}, {method.lower()} until golden, plate with sides." if p_str else None,
        f"Combine all ingredients, cook until done, serve hot.",
    ]
    return next((t for t in templates if t), "Combine ingredients and cook until done.")


def generate_recipe_name(method, protein, vegetables, grain, dish_type, cuisine_adj):
    """Create a recipe name from components."""
    parts = []
    adj = random.choice(cuisine_adj) if random.random() < 0.3 else ""
    if adj:
        parts.append(adj)

    parts.append(method)

    if protein:
        parts.append(protein.replace("_", " ").title())
    elif vegetables:
        parts.append(vegetables[0].replace("_", " ").title())

    if dish_type and random.random() < 0.6:
        parts.append(dish_type)
    elif vegetables and protein:
        parts.append("with " + vegetables[0].replace("_", " ").title())

    return " ".join(parts)


def generate_recipes():
    """Generate 100,000+ recipes."""
    recipes = []
    seen_names = set()
    target = 105000

    for cuisine_name, profile in CUISINES.items():
        per_cuisine = target // len(CUISINES) + 1000  # extra buffer

        for meal_type in ["breakfast", "lunch", "dinner"]:
            rules = MEAL_TYPE_RULES[meal_type]
            per_slot = per_cuisine // 3 + 200
            generated = 0

            # Determine ingredient pools for this cuisine+meal
            proteins = profile["proteins"]
            if rules["preferred_proteins"]:
                proteins = [p for p in proteins if p in rules["preferred_proteins"]] or proteins[:3]

            vegs = profile["vegetables"]
            grains = profile.get("grains", [])
            dairy_pool = profile.get("dairy", [])
            spice_pool = profile.get("spices", ["salt", "pepper"])
            sauce_pool = profile.get("sauces", [])
            legume_pool = profile.get("legumes", [])

            methods = profile["methods"]
            dishes = profile["dishes"]

            attempts = 0
            max_attempts = per_slot * 4

            while generated < per_slot and attempts < max_attempts:
                attempts += 1

                # Pick components
                method = random.choice(methods)
                dish_type = random.choice(dishes)

                ingredients = []

                # Protein (80% chance for lunch/dinner, 60% for breakfast)
                protein = None
                p_chance = 0.6 if meal_type == "breakfast" else 0.8
                if proteins and random.random() < p_chance:
                    protein = random.choice(proteins)
                    ingredients.append(protein)

                # Vegetables (1-3)
                n_veg = random.randint(1, min(4, len(vegs)))
                chosen_vegs = random.sample(vegs, min(n_veg, len(vegs)))
                ingredients.extend(chosen_vegs)

                # Grain (50% chance)
                grain = None
                if grains and random.random() < 0.5:
                    grain = random.choice(grains)
                    ingredients.append(grain)

                # Dairy (40% chance)
                if dairy_pool and random.random() < 0.4:
                    ingredients.append(random.choice(dairy_pool))

                # Legume (20% chance, more for veg meals)
                if legume_pool and random.random() < (0.4 if not protein else 0.15):
                    ingredients.append(random.choice(legume_pool))

                # Spice (1-2)
                n_spice = random.randint(1, min(3, len(spice_pool)))
                chosen_spices = random.sample(spice_pool, min(n_spice, len(spice_pool)))
                ingredients.extend(chosen_spices)

                # Sauce (0-1)
                if sauce_pool and random.random() < 0.5:
                    ingredients.append(random.choice(sauce_pool))

                # Deduplicate and enforce max
                ingredients = list(dict.fromkeys(ingredients))
                max_ing = rules["max_ingredients"]
                if len(ingredients) > max_ing:
                    # Keep protein + first veg + spices, trim rest
                    ingredients = ingredients[:max_ing]

                if len(ingredients) < 3:
                    continue

                # Generate name
                name = generate_recipe_name(
                    method, protein, chosen_vegs, grain, dish_type,
                    profile["adj"]
                )

                # Ensure unique names
                if name in seen_names:
                    # Add a variation number
                    for suffix in range(2, 50):
                        variant = f"{name} #{suffix}"
                        if variant not in seen_names:
                            name = variant
                            break
                    else:
                        continue

                if name in seen_names:
                    continue
                seen_names.add(name)

                # Generate rest
                tags = generate_tags(ingredients, profile.get("tags", []))
                calories, protein_g, carbs, fat = estimate_nutrition(ingredients, meal_type)
                instruction = generate_instruction(method, ingredients, dish_type)
                ing_str = "|".join(ingredients)

                recipes.append({
                    "title": name,
                    "meal_type": meal_type,
                    "cuisine": cuisine_name,
                    "tags": tags,
                    "ingredients": ing_str,
                    "instructions": instruction,
                    "calories": calories,
                    "protein": protein_g,
                    "carbs": carbs,
                    "fat": fat,
                })
                generated += 1

    return recipes


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "recipes.csv")
    out_path = os.path.normpath(out_path)

    print(f"Generating recipes...")
    recipes = generate_recipes()
    random.shuffle(recipes)

    print(f"Generated {len(recipes):,} recipes. Writing to {out_path}...")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "title", "meal_type", "cuisine", "tags", "ingredients",
            "instructions", "calories", "protein", "carbs", "fat"
        ])
        writer.writeheader()
        writer.writerows(recipes)

    file_size = os.path.getsize(out_path)
    print(f"Done! {len(recipes):,} recipes written ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()

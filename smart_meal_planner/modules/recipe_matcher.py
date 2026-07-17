"""
Recipe Matching Module
======================
Ranks recipes from the corpus against a user's available ingredients.

Primary sort: fewest missing ingredients first (0 missing = cook right now).
Secondary   : highest coverage %.
Tiebreaker  : TF-IDF cosine similarity.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_RECIPES_FILE = _DATA_DIR / "recipes.csv"


class RecipeMatcher:
    """Content-based recipe recommender — ingredient coverage is king."""

    # Basic pantry staples are ALWAYS assumed available — never counted as missing
    PANTRY_STAPLES = frozenset({
        "salt", "pepper", "water", "oil", "vegetable_oil", "olive_oil",
        "sugar", "black_pepper", "cooking_spray", "butter", "flour",
        "garlic", "onion", "baking_powder", "baking_soda", "vinegar",
        "soy_sauce", "tomato_paste", "cumin", "paprika", "turmeric",
    })

    def __init__(
        self,
        alpha: float = 0.15,         # TF-IDF weight
        beta: float = 0.85,          # Coverage weight (dominant signal)
        gamma: float = 0.05,         # Dietary tag bonus
        min_coverage: float = 0.35,  # Lowered: more recipes surface, still relevant
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_coverage = min_coverage

        self._df = None
        self._vectorizer = None
        self._recipe_matrix = None
        self._unique_ingredients = set()
        self._load_and_fit()

    # ── Loading & fitting ──────────────────────────────────────────────────────

    def _load_and_fit(self) -> None:
        """Load corpus, build TF-IDF and unique ingredients set, and cache the result."""
        cache_file = _DATA_DIR / "matcher_cache.pkl"
        
        # Check if cache exists and is newer than recipes.csv
        use_cache = False
        if cache_file.exists() and _RECIPES_FILE.exists():
            if os.path.getmtime(cache_file) > os.path.getmtime(_RECIPES_FILE):
                use_cache = True
                
        if use_cache:
            try:
                logger.info("Loading RecipeMatcher from cache...")
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self._df = data["df"]
                    self._vectorizer = data["vectorizer"]
                    self._recipe_matrix = data["matrix"]
                    self._unique_ingredients = data["unique_ingredients"]
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding...")

        logger.info("Building RecipeMatcher from scratch...")
        if not _RECIPES_FILE.exists():
            raise FileNotFoundError(f"Recipe corpus not found: {_RECIPES_FILE}")
            
        df = pd.read_csv(_RECIPES_FILE)
        df["ingredients"] = df["ingredients"].str.lower().str.replace(r"\s+", "_", regex=True)
        df["tags"] = df["tags"].fillna("").str.lower()
        df["ingredient_list"] = df["ingredients"].apply(
            lambda s: sorted([i.strip() for i in s.split("|") if i.strip()])
        )
        self._df = df
        
        self._unique_ingredients = set()
        for ing_set in df["ingredient_list"]:
            self._unique_ingredients.update(ing_set)

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
        )
        corpus_docs = self._df["ingredients"].str.replace("|", " ").tolist()
        self._vectorizer.fit(corpus_docs)
        self._recipe_matrix = self._vectorizer.transform(corpus_docs)
        logger.info("TF-IDF fitted on %d recipes.", len(self._df))
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({
                    "df": self._df,
                    "vectorizer": self._vectorizer,
                    "matrix": self._recipe_matrix,
                    "unique_ingredients": self._unique_ingredients
                }, f)
            logger.info("Saved RecipeMatcher cache.")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    # ── Ingredient matching helpers ────────────────────────────────────────────

    @staticmethod
    def _normalize_ingredient(name: str) -> str:
        return name.strip().lower().replace(" ", "_")

    @classmethod
    def _ingredient_matches(cls, corpus_ing: str, user_ing: str) -> bool:
        """Match corpus and user ingredients without loose substring false positives."""
        if corpus_ing == user_ing:
            return True
        c_parts = corpus_ing.split("_")
        u_parts = user_ing.split("_")
        if user_ing in c_parts or corpus_ing in u_parts:
            return True
        if len(c_parts[0]) > 3 and c_parts[0] == u_parts[0]:
            return True
        for a, b in ((user_ing, corpus_ing), (corpus_ing, user_ing)):
            if len(a) < 4:
                continue
            if a not in b:
                continue
            idx = b.find(a)
            before = idx == 0 or b[idx - 1] == "_"
            after = idx + len(a) == len(b) or b[idx + len(a)] == "_"
            if before and after:
                return True
        return False

    def _build_expanded_user_set(self, user_set: Set[str]) -> Set[str]:
        expanded = set(user_set)
        for ing in user_set:
            parts = ing.split("_")
            for part in parts:
                if len(part) > 3:
                    expanded.add(part)
            if parts:
                expanded.add(parts[0])
        return expanded

    def _build_matched_ingredients(
        self,
        user_set: Set[str],
        extra_ingredients: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Return corpus ingredient names considered available to the user."""
        expanded = self._build_expanded_user_set(user_set)
        search_space = self._unique_ingredients
        if extra_ingredients:
            search_space = search_space | extra_ingredients

        matched: Set[str] = set()
        for ing in search_space:
            if ing in self.PANTRY_STAPLES:
                matched.add(ing)
                continue
            if (
                ing in user_set
                or ing in expanded
                or ing.split("_")[0] in expanded
            ):
                matched.add(ing)
                continue
            for user_ing in user_set:
                if self._ingredient_matches(ing, user_ing):
                    matched.add(ing)
                    break
        return matched

    @classmethod
    def _leftover_used(cls, recipe_ing: str, leftover: str) -> bool:
        return cls._ingredient_matches(recipe_ing, leftover)

    # ── Public API ─────────────────────────────────────────────────────────────

    def match(
        self,
        user_ingredients: List[str],
        meal_type: Optional[str] = None,
        dietary_filters: Optional[List[str]] = None,
        top_n: int = 10,
        exclude_titles: Optional[Set[str]] = None,
        strict_mode: bool = False,
        indian_focus: bool = False,
    ) -> pd.DataFrame:
        """
        Rank recipes by ingredient match.

        Sorting priority:
          1. Fewest missing ingredients  (0 missing = can cook RIGHT NOW ✅)
          2. Highest coverage %
          3. TF-IDF similarity as tiebreaker

        Parameters
        ----------
        user_ingredients : canonical ingredient names the user has
        meal_type        : 'breakfast' | 'lunch' | 'dinner' | None
        dietary_filters  : required tags e.g. ['vegetarian', 'gluten-free']
        top_n            : max results to return
        exclude_titles   : already-used recipe titles (no-repeat)
        strict_mode      : True → only return recipes with 0 missing ingredients

        Returns
        -------
        pd.DataFrame sorted by (missing_count ASC, coverage DESC, final_score DESC)
        """
        if not user_ingredients:
            return pd.DataFrame()

        user_set = set(user_ingredients)

        # ── Apply hard filters (mask on index — avoid copying full corpus) ─────
        mask = pd.Series(True, index=self._df.index)
        if meal_type:
            mask &= self._df["meal_type"].str.lower() == meal_type.lower()
        if dietary_filters:
            for tag in dietary_filters:
                mask &= self._df["tags"].str.contains(tag.lower(), na=False)
        if exclude_titles:
            mask &= ~self._df["title"].isin(exclude_titles)
        if not mask.any():
            return pd.DataFrame()

        df = self._df.loc[mask]

        # ── Compute coverage & missing count for every recipe ─────────────────
        coverage_list = []
        missing_count_list = []
        missing_ing_list = []

        matched_corpus_ingredients = self._build_matched_ingredients(user_set)

        for ing_set in df["ingredient_list"]:
            if not ing_set:
                coverage_list.append(0.0)
                missing_count_list.append(0)
                missing_ing_list.append([])
                continue

            have_count = 0
            missing = []
            for i in ing_set:
                if i in matched_corpus_ingredients:
                    have_count += 1
                else:
                    missing.append(i)

            coverage_list.append(have_count / len(ing_set))
            missing_count_list.append(len(missing))
            missing_ing_list.append(missing)

        coverage_arr = np.array(coverage_list)
        missing_arr  = np.array(missing_count_list, dtype=int)

        # ── TF-IDF cosine similarity (used as tiebreaker) ─────────────────────
        user_doc    = " ".join(user_ingredients)
        user_vec    = self._vectorizer.transform([user_doc])
        recipe_vecs = self._recipe_matrix[df.index]
        cosine_arr  = cosine_similarity(user_vec, recipe_vecs).flatten()

        # ── Tag bonus ─────────────────────────────────────────────────────────
        tag_bonus = np.zeros(len(df))
        if dietary_filters:
            for i, tags in enumerate(df["tags"].values):
                matched = sum(1 for t in dietary_filters if t in tags)
                tag_bonus[i] = matched / max(len(dietary_filters), 1)

        # ── Build coverage mask ───────────────────────────────────────────────
        if strict_mode:
            # STRICT: only recipes where the user has every single ingredient
            mask = missing_arr == 0
            if mask.sum() == 0:
                logger.info("strict_mode: no perfect matches for %s, relaxing to 60%%.", meal_type)
                strict_mode = False

        if not strict_mode:
            # Progressive relaxation: start at min_coverage, lower if needed
            for threshold in (self.min_coverage, 0.40, 0.30, 0.20):
                mask = coverage_arr >= threshold
                if mask.sum() > 0:
                    if threshold < self.min_coverage:
                        logger.info(
                            "Coverage relaxed to %.0f%% for %s (no results at %.0f%%).",
                            threshold * 100, meal_type, self.min_coverage * 100,
                        )
                    break

        # ── Apply mask ────────────────────────────────────────────────────────
        df_f           = df[mask].copy()
        coverage_f     = coverage_arr[mask]
        missing_f      = missing_arr[mask]
        cosine_f       = cosine_arr[mask]
        tag_bonus_f    = tag_bonus[mask]
        missing_ings_f = [missing_ing_list[i] for i, m in enumerate(mask) if m]

        if df_f.empty:
            return pd.DataFrame()

        # ── Score: heavier missing penalty so 0-missing always wins ───────────
        # Each missing ingredient = -0.08 penalty (capped at -0.40)
        # Scaled by recipe size so large recipes aren't unfairly penalized
        recipe_sizes = np.array([len(il) for il in df_f["ingredient_list"].values], dtype=float)
        recipe_sizes = np.maximum(recipe_sizes, 1.0)  # avoid div/0
        relative_missing = missing_f / recipe_sizes   # fraction of recipe that's missing
        missing_penalty = np.clip(relative_missing * 0.50, 0.0, 0.45)

        final_scores = (
            self.beta  * coverage_f
            + self.alpha * cosine_f
            + self.gamma * tag_bonus_f
            - missing_penalty
        )

        is_indian = np.zeros(len(df_f))
        if indian_focus:
            cuisine_series = df_f["cuisine"].fillna("").str.lower()
            is_indian = (cuisine_series == "indian").astype(float)
            final_scores = final_scores + 0.5 * is_indian

        result = df_f.copy()
        result["tfidf_score"]         = cosine_f
        result["coverage"]            = coverage_f
        result["missing_count"]       = missing_f
        result["tag_bonus"]           = tag_bonus_f
        result["final_score"]         = final_scores
        result["missing_ingredients"] = missing_ings_f
        result["is_indian"]           = is_indian

        # ── Sort: is_indian DESC (if focus on), then 0 missing FIRST, then coverage DESC, then final_score DESC
        sort_keys = ["missing_count", "coverage", "final_score"]
        sort_ascending = [True, False, False]
        if indian_focus:
            sort_keys.insert(0, "is_indian")
            sort_ascending.insert(0, False)

        result = (
            result
            .sort_values(
                sort_keys,
                ascending=sort_ascending,
            )
            .head(top_n)
            .reset_index(drop=True)
        )
        return result

    def get_recipe_by_title(self, title: str):
        """Fetch a single recipe row by exact title."""
        rows = self._df[self._df["title"] == title]
        return rows.iloc[0] if not rows.empty else None

    def match_leftovers(
        self,
        leftover_items: List[str],
        available_pantry: Optional[List[str]] = None,
        dietary_filters: Optional[List[str]] = None,
        top_n: int = 10,
        indian_focus: bool = False,
    ) -> pd.DataFrame:
        """
        Rank recipes by how many leftover items they use.
        Leftovers are counted as a first-priority signal (must-use items),
        then coverage by pantry fills the rest.

        Parameters
        ----------
        leftover_items    : items the user must use (e.g. ['cooked chicken', 'spinach'])
        available_pantry  : other ingredients the user has on hand
        dietary_filters   : optional dietary constraint tags
        top_n             : max results

        Returns
        -------
        pd.DataFrame sorted by leftover coverage DESC, overall coverage DESC
        """
        if not leftover_items:
            return pd.DataFrame()

        leftover_set = {self._normalize_ingredient(i) for i in leftover_items}
        all_user = set(leftover_set)
        if available_pantry:
            all_user.update(self._normalize_ingredient(i) for i in available_pantry)
        all_user.update(self.PANTRY_STAPLES)

        mask = pd.Series(True, index=self._df.index)
        if dietary_filters:
            for tag in dietary_filters:
                mask &= self._df["tags"].str.contains(tag.lower(), na=False)
        if not mask.any():
            return pd.DataFrame()

        df = self._df.loc[mask]

        leftover_coverage = []
        overall_coverage = []
        missing_ings_list = []

        recipe_ingredients: Set[str] = set()
        for ing_set in df["ingredient_list"]:
            recipe_ingredients.update(ing_set)

        matched_ingredients = self._build_matched_ingredients(all_user, recipe_ingredients)

        for ing_set in df["ingredient_list"]:
            if not ing_set:
                leftover_coverage.append(0.0)
                overall_coverage.append(0.0)
                missing_ings_list.append([])
                continue

            have_count = 0
            used_leftovers: Set[str] = set()
            missing = []

            for i in ing_set:
                if i in matched_ingredients:
                    have_count += 1
                    for lft in leftover_set:
                        if self._leftover_used(i, lft):
                            used_leftovers.add(lft)
                else:
                    missing.append(i)

            n = len(ing_set)
            leftover_coverage.append(len(used_leftovers) / max(len(leftover_set), 1))
            overall_coverage.append(have_count / n if n else 0.0)
            missing_ings_list.append(missing)
        lc_arr = np.array(leftover_coverage)
        oc_arr = np.array(overall_coverage)

        # Must use at least one leftover item
        mask = lc_arr > 0
        if mask.sum() == 0:
            return pd.DataFrame()

        df_f = df[mask].copy()
        lc_f = lc_arr[mask]
        oc_f = oc_arr[mask]
        missing_f = [missing_ings_list[i] for i, m in enumerate(mask) if m]

        if df_f.empty:
            return pd.DataFrame()

        # Score: leftover coverage dominates (0.7) then overall (0.3)
        final_scores = 0.7 * lc_f + 0.3 * oc_f

        is_indian = np.zeros(len(df_f))
        if indian_focus:
            cuisine_series = df_f["cuisine"].fillna("").str.lower()
            is_indian = (cuisine_series == "indian").astype(float)
            final_scores = final_scores + 0.5 * is_indian

        df_f["leftover_coverage"] = lc_f
        df_f["coverage"] = oc_f
        df_f["final_score"] = final_scores
        df_f["missing_ingredients"] = missing_f
        df_f["is_indian"] = is_indian

        sort_keys = ["leftover_coverage", "coverage", "final_score"]
        sort_ascending = [False, False, False]
        if indian_focus:
            sort_keys.insert(0, "is_indian")
            sort_ascending.insert(0, False)

        return (
            df_f
            .sort_values(sort_keys,
                         ascending=sort_ascending)
            .head(top_n)
            .reset_index(drop=True)
        )

    @property
    def all_titles(self) -> List[str]:
        return self._df["title"].tolist()

    @property
    def corpus(self) -> pd.DataFrame:
        return self._df

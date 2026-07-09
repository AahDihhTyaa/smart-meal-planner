"""
NLP Normalization Module
========================
Turns raw ingredient strings into a clean, canonical, deduplicated list.

Pipeline:
  1. Lowercasing
  2. Quantity & unit stripping  (regex)
  3. Tokenization & lemmatization (spaCy, with NLTK fallback)
  4. Stopword removal
  5. Synonym / alias mapping  (ingredient_synonyms.json)
  6. Fuzzy deduplication (RapidFuzz)
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_SYNONYMS_FILE = _DATA_DIR / "ingredient_synonyms.json"

# ── Unit & quantity patterns ──────────────────────────────────────────────────
_UNIT_PATTERN = re.compile(
    r"""
    ^\s*
    (?:                         # optional leading fraction/decimal/integer
        \d+\s*[-/]\s*\d+        # e.g. 1/2
        | \d+\.?\d*             # e.g. 2, 2.5
        | [½⅓⅔¼¾⅛⅜⅝⅞]          # unicode fractions
    )?
    \s*
    (?:                         # optional unit
        tsp|tbsp|tablespoon|teaspoon
        |cup|cups
        |oz|ounce|ounces
        |lb|pound|pounds
        |g|gram|grams|kg|kilogram|kilograms
        |ml|milliliter|milliliters|liter|liters|l
        |handful|handfuls
        |bunch|bunches
        |clove|cloves
        |slice|slices
        |piece|pieces|pcs?
        |small|medium|large
        |can|cans
        |package|packages|pkg
        |head|heads
        |inch|inches
        |sprig|sprigs
        |pinch|pinches
        |dash|dashes
    )?
    \s+
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Stopwords to strip after lemmatization — extended set
_STOPWORDS = frozenset(
    [
        "of", "the", "a", "an", "some", "fresh", "frozen", "dried", "raw",
        "cooked", "chopped", "diced", "minced", "sliced", "grated", "shredded",
        "peeled", "whole", "halved", "crushed", "ground", "powdered", "mixed",
        "optional", "and", "or", "with", "to", "taste", "finely", "roughly",
        "thinly", "thick", "thin", "ripe", "washed", "pitted", "seeded",
        "deseeded", "trimmed", "cleaned", "prepared", "ready", "coarsely",
        "loosely", "packed", "heaped", "level", "sifted", "roasted", "toasted",
        "blanched", "steamed", "boiled", "fried", "baked", "grilled", "smoked",
        "canned", "tinned", "jarred", "bottled", "organic", "free", "range",
        "unsweetened", "sweetened",
    ]
)


class NLPNormalizer:
    """Normalizes raw ingredient strings into canonical token sets."""

    def __init__(self, fuzzy_threshold: int = 88):
        """
        Parameters
        ----------
        fuzzy_threshold : int
            RapidFuzz similarity score (0-100) below which two ingredients are
            considered different. 88 balances typo tolerance vs false merges.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self._synonyms: Dict[str, str] = self._load_synonyms()
        self._nlp = self._load_spacy()
        self._lemmatizer = self._load_lemmatizer()

    # ── Loading helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _load_synonyms() -> Dict[str, str]:
        if _SYNONYMS_FILE.exists():
            with _SYNONYMS_FILE.open(encoding="utf-8") as f:
                return json.load(f)
        logger.warning("Synonyms file not found: %s", _SYNONYMS_FILE)
        return {}

    @staticmethod
    def _load_lemmatizer():
        """Create NLTK lemmatizer once; spaCy path does not use this."""
        try:
            from nltk.stem import WordNetLemmatizer  # noqa: PLC0415
            import nltk  # noqa: PLC0415

            try:
                return WordNetLemmatizer()
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                return WordNetLemmatizer()
        except ImportError:
            return None

    @staticmethod
    def _load_spacy():
        """Load spaCy model; fall back gracefully if unavailable."""
        try:
            import spacy  # noqa: PLC0415

            try:
                return spacy.load("en_core_web_sm")
            except OSError:
                # Model not downloaded yet — use a blank English model for
                # tokenization only (no lemmatization vectors, but still useful)
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                return spacy.blank("en")
        except ImportError:
            logger.warning("spaCy not installed. Falling back to basic tokenizer.")
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(self, raw: str) -> Optional[str]:
        """
        Normalize a single ingredient string → canonical name or None.

        Examples
        --------
        >>> n = NLPNormalizer()
        >>> n.normalize("2 cups of diced tomatoes")
        'tomato'
        >>> n.normalize("scallions")
        'green_onion'
        """
        text = raw.strip().lower()
        if not text:
            return None

        # 1. Strip quantity & units
        text = _UNIT_PATTERN.sub("", text).strip()
        if not text:
            return None

        # 2. Synonym lookup (before lemmatization, on raw lowercased text)
        canonical = self._synonyms.get(text)
        if canonical:
            return canonical

        # 3. Lemmatize
        text = self._lemmatize(text)

        # 4. Remove stopwords & punctuation
        tokens = [
            t for t in re.split(r"[\s,]+", text) if t and t not in _STOPWORDS
        ]
        if not tokens:
            return None
        text = "_".join(tokens)  # canonical multi-word → underscore

        # 5. Synonym lookup again after lemmatization
        canonical = self._synonyms.get(text)
        if canonical:
            return canonical

        # 5b. Try space-separated version (in case underscores differ from JSON keys)
        space_text = text.replace("_", " ")
        canonical = self._synonyms.get(space_text)
        if canonical:
            return canonical

        # 6. Single-word synonym lookup on each token
        canonical_tokens = []
        for tok in tokens:
            mapped = self._synonyms.get(tok, tok)
            # Also try without trailing 's' for plurals not caught by lemmatizer
            if mapped == tok and tok.endswith("s") and len(tok) > 3:
                mapped = self._synonyms.get(tok[:-1], tok)
            canonical_tokens.append(mapped)
        text = "_".join(canonical_tokens)

        return text if text else None

    def normalize_list(self, raw_ingredients: List[str]) -> Dict[str, str]:
        """
        Normalize a list of raw strings.

        Returns
        -------
        dict : {canonical_name → original_text}  (deduplicated)
        """
        seen: Dict[str, str] = {}
        for raw in raw_ingredients:
            canonical = self.normalize(raw)
            if canonical:
                seen[canonical] = raw
        return self._fuzzy_deduplicate(seen)

    def parse_text_input(self, text: str) -> List[str]:
        """
        Split a free-text ingredient block into individual raw strings.

        Handles comma-separated, newline-separated, and numbered lists.
        """
        # Remove list markers like "1.", "2)", "-", "•"
        text = re.sub(r"^\s*[\d]+[.)]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-•*]\s*", "", text, flags=re.MULTILINE)

        # Split on commas and newlines
        parts = re.split(r"[,\n;]+", text)
        return [p.strip() for p in parts if p.strip()]

    def get_ingredient_set(self, raw_text: str) -> List[str]:
        """
        Full pipeline: raw text → deduplicated canonical ingredient list.

        This is the main entry point for the Streamlit UI.
        """
        raw_list = self.parse_text_input(raw_text)
        normalized = self.normalize_list(raw_list)
        return sorted(normalized.keys())

    # ── Private helpers ────────────────────────────────────────────────────────

    def _lemmatize(self, text: str) -> str:
        """Return lemmatized text using spaCy if available, else NLTK/identity."""
        if self._nlp is not None:
            doc = self._nlp(text)
            # If the model has lemmatizer, use it; else fall back to token text
            lemmas = []
            for token in doc:
                lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
                lemmas.append(lemma)
            return " ".join(lemmas)
        # NLTK fallback
        if self._lemmatizer is not None:
            tokens = text.split()
            return " ".join(self._lemmatizer.lemmatize(t) for t in tokens)
        return text

    def _fuzzy_deduplicate(self, seen: Dict[str, str]) -> Dict[str, str]:
        """
        Merge near-duplicate keys (e.g. 'tomatoe' vs 'tomato').

        The first-seen (alphabetically earlier) key wins.
        """
        keys = sorted(seen.keys())
        merged: Dict[str, str] = {}
        absorbed = set()

        for i, key in enumerate(keys):
            if key in absorbed:
                continue
            merged[key] = seen[key]
            # Compare against remaining keys
            for other in keys[i + 1 :]:
                if other in absorbed:
                    continue
                score = fuzz.ratio(key, other)
                if score >= self.fuzzy_threshold:
                    absorbed.add(other)  # merge other → key

        return merged

    # ── Quantity extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_quantity(raw: str) -> Tuple[Optional[str], str]:
        """
        Extract leading quantity+unit from a string.

        Returns
        -------
        (quantity_str, remainder)

        Examples
        --------
        >>> NLPNormalizer.extract_quantity("2 cups of flour")
        ('2 cups', 'flour')
        """
        match = _UNIT_PATTERN.match(raw.lower())
        if match:
            qty = raw[: match.end()].strip()
            remainder = raw[match.end() :].strip()
            return (qty if qty else None, remainder)
        return (None, raw.strip())

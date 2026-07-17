"""
Vision Input Module
===================
Sends an image to a Vision AI API and returns a structured ingredient list.

Supports two free providers (auto-detected from env vars):
  1. Groq  (GROQ_API_KEY)   — fast, free, no credit card
  2. Gemini (GEMINI_API_KEY) — free tier via Google AI Studio

Falls back gracefully (returns empty list + error message) if no key is set.
"""

import base64
import json
import logging
import os
import re
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Primary prompt — highly detailed for maximum accuracy ─────────────────────
_PROMPT = """
You are an expert food ingredient recognition system with professional culinary knowledge.

Carefully analyze this image and identify ALL food ingredients, groceries, produce, 
and packaged food items visible.

INSTRUCTIONS:
- Scan the ENTIRE image systematically (left-to-right, top-to-bottom)
- Identify both whole and partial items visible
- Read any visible text labels or packaging to get exact product names
- For produce: identify variety if possible (e.g. "roma tomato" not just "tomato")
- For packaged goods: use the product type (e.g. "coconut milk" not "can")
- Estimate quantities from visual count or package size
- Include spices, condiments, sauces, oils if visible
- CRITICAL: Do NOT stop early. Many images contain 12-20+ distinct ingredients. You MUST list EVERY single distinct food item you can find.

Return ONLY a valid JSON array with no markdown, no explanation, no extra text.
Format exactly:
[
  {"name": "chicken breast", "quantity": "2", "confidence": "high"},
  {"name": "garlic", "quantity": "1 head", "confidence": "high"},
  {"name": "olive oil", "quantity": "1 bottle", "confidence": "medium"}
]

Confidence levels:
- "high"   = clearly visible and identifiable
- "medium" = visible but partially obscured or partially legible label
- "low"    = inferred / uncertain

RULES:
- Use simple lowercase English ingredient names
- Do NOT include non-food items (utensils, plates, packaging materials)
- Do NOT hallucinate items you cannot actually see
- If no food is visible, return an empty array: []
- Minimum: return every item you can see with at least "low" confidence
""".strip()

# ── Fallback prompt — simpler, for retry when primary fails ───────────────────
_FALLBACK_PROMPT = """
List all food items visible in this image.
Return ONLY a JSON array like: [{"name": "tomato", "quantity": "3"}]
No markdown, no extra text. Just the JSON array.
""".strip()

# ── Barcode / Nutrition Label prompt ─────────────────────────────────────────
_BARCODE_PROMPT = """
You are an expert food label and barcode reader with nutritional science knowledge.

Carefully analyze this image of a packaged food product — it may show a nutrition facts panel, product label, or barcode.

Extract and return ONLY a valid JSON object with this exact structure:
{
  "product_name": "Product or brand name",
  "brand": "Brand name or empty string",
  "serving_size": "e.g. 100g or 1 cup",
  "servings_per_container": "number or unknown",
  "calories_per_serving": 0,
  "protein_g": 0.0,
  "carbs_g": 0.0,
  "sugar_g": 0.0,
  "fat_g": 0.0,
  "saturated_fat_g": 0.0,
  "sodium_mg": 0.0,
  "fiber_g": 0.0,
  "ingredient_list": ["list", "of", "main", "ingredients"],
  "allergens": ["allergen1", "allergen2"],
  "confidence": "high | medium | low"
}

RULES:
- If a field is not visible, use 0 for numbers or empty string/list for text/arrays.
- Round all numbers to 1 decimal place.
- Extract visible ingredients from the ingredient list text if present.
- confidence = "high" if nutrition facts panel clearly visible, "medium" if partial, "low" if only brand visible.
- Return ONLY the raw JSON. No markdown, no explanation.
""".strip()


class VisionInputModule:
    """Wraps a Vision AI API for ingredient detection from photos.

    Tries providers in order: Groq → Gemini.
    Uses gemini-2.0-flash for best accuracy when Gemini key is set.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
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
            if gemini_key:
                self._init_gemini(gemini_key)
            if not self._available and openai_key:
                self._init_openai(openai_key)
            if not self._available and groq_key:
                self._init_groq(groq_key)

    def _init_openai(self, key: str) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
            self._provider = "openai"
            self._available = True
            logger.info("Vision: using OpenAI (gpt-4o).")
        except ImportError:
            logger.warning("openai package not installed.")
        except Exception as e:
            logger.warning("OpenAI init failed: %s", e)

    # ── Provider init ──────────────────────────────────────────────────────────

    def _init_groq(self, key: str) -> None:
        try:
            from groq import Groq  # noqa: PLC0415
            self._client = Groq(api_key=key)
            self._provider = "groq"
            self._available = True
            logger.info("Vision: using Groq (llama-4-scout).")
        except ImportError:
            logger.warning("groq package not installed.")
        except Exception as e:
            logger.warning("Groq init failed: %s", e)

    def _init_gemini(self, key: str) -> None:
        try:
            from google import genai  # noqa: PLC0415
            self._client = genai.Client(api_key=key)
            self._provider = "gemini"
            self._available = True
            logger.info("Vision: using Gemini (gemini-2.0-flash).")
        except ImportError:
            logger.warning("google-genai not installed.")
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def provider_name(self) -> str:
        return self._provider or "none"

    def detect_ingredients(
        self, image_bytes: bytes, mime_type: str = "image/jpeg",
        min_confidence: str = "low"
    ) -> Tuple[List[dict], Optional[str]]:
        """
        Detect ingredients from image bytes with two-pass retry logic.

        Parameters
        ----------
        image_bytes    : raw image bytes
        mime_type      : MIME type of the image
        min_confidence : filter results below this confidence ('low'|'medium'|'high')

        Returns (ingredient_list, error_message).
        """
        if not self._available:
            return [], (
                "Vision API not configured. "
                "Add a GROQ_API_KEY or GEMINI_API_KEY in the sidebar."
            )

        # First pass — full detailed prompt
        if self._provider == "openai":
            results, err = self._detect_openai(image_bytes, mime_type, _PROMPT)
        elif self._provider == "groq":
            results, err = self._detect_groq(image_bytes, mime_type, _PROMPT)
        else:
            results, err = self._detect_gemini(image_bytes, mime_type, _PROMPT)

        # Second pass — fallback simple prompt if first pass failed or empty
        if (err or not results):
            logger.info("Primary detection failed/empty, retrying with fallback prompt.")
            if self._provider == "openai":
                results, err = self._detect_openai(image_bytes, mime_type, _FALLBACK_PROMPT)
            elif self._provider == "groq":
                results, err = self._detect_groq(image_bytes, mime_type, _FALLBACK_PROMPT)
            else:
                results, err = self._detect_gemini(image_bytes, mime_type, _FALLBACK_PROMPT)

        # Filter by confidence
        if results and min_confidence != "low":
            conf_rank = {"high": 3, "medium": 2, "low": 1}
            min_rank = conf_rank.get(min_confidence, 1)
            results = [
                r for r in results
                if conf_rank.get(r.get("confidence", "low"), 1) >= min_rank
            ]

        return results, err

    # ── Provider implementations ───────────────────────────────────────────────

    def _detect_openai(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> Tuple[List[dict], Optional[str]]:
        try:
            b64 = base64.b64encode(image_bytes).decode()
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.05,
                max_tokens=2048,
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_response(raw_text)
        except Exception as e:
            logger.error("OpenAI vision error: %s", e)
            return [], f"OpenAI error: {e}"

    def _detect_groq(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> Tuple[List[dict], Optional[str]]:
        try:
            b64 = base64.b64encode(image_bytes).decode()
            response = self._client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.05,    # near-deterministic for consistency
                max_tokens=2048,     # increased for large ingredient lists
            )
            raw_text = response.choices[0].message.content.strip()
            return self._parse_response(raw_text)
        except Exception as e:
            logger.error("Groq vision error: %s", e)
            return [], f"Groq error: {e}"

    def _detect_gemini(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> Tuple[List[dict], Optional[str]]:
        """
        Try multiple Gemini models in order, with automatic retry on 429 errors.
        Model priority: gemini-2.5-flash-lite → gemini-1.5-flash → gemini-2.0-flash
        """
        from google.genai import types  # noqa: PLC0415

        # Models to try in order (lite/faster models first to preserve quota)
        GEMINI_MODELS = [
            "gemini-2.5-flash-lite",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]

        last_error = None
        for model in GEMINI_MODELS:
            for attempt in range(3):  # up to 3 retries per model
                try:
                    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    response = self._client.models.generate_content(
                        model=model,
                        contents=[prompt, image_part],
                        config=types.GenerateContentConfig(
                            temperature=0.05,
                            max_output_tokens=2048,
                        ),
                    )
                    raw_text = response.text.strip()
                    logger.info("Gemini vision succeeded with model: %s", model)
                    return self._parse_response(raw_text)

                except Exception as e:
                    err_str = str(e)
                    last_error = err_str

                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        # Extract retry delay from error message if available
                        import re as _re
                        delay_match = _re.search(r"retry.*?(\d+)s", err_str, _re.IGNORECASE)
                        wait_sec = int(delay_match.group(1)) if delay_match else (10 * (attempt + 1))
                        wait_sec = min(wait_sec, 30)  # cap at 30s
                        logger.warning(
                            "Gemini 429 on model %s (attempt %d). Waiting %ds then trying next model.",
                            model, attempt + 1, wait_sec
                        )
                        time.sleep(wait_sec)
                        break  # move to the next model immediately
                    else:
                        # Non-quota error — log and try next model
                        logger.error("Gemini error on model %s: %s", model, e)
                        break  # don't retry non-quota errors

        return [], f"Gemini error: All models exhausted. Last error: {last_error}"

    def _parse_response(self, raw_text: str) -> Tuple[List[dict], Optional[str]]:
        """
        Parse a JSON ingredient list from the model response.
        Uses multiple strategies to extract valid JSON even from messy output.
        """
        # Strategy 1: strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Strategy 2: try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return self._clean_items(parsed), None
        except json.JSONDecodeError:
            pass

        # Strategy 3: extract first JSON array with regex
        array_match = re.search(r"\[.*?\]", text, re.DOTALL)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    return self._clean_items(parsed), None
            except json.JSONDecodeError:
                pass

        # Strategy 4: extract individual JSON objects and wrap them
        objects = re.findall(r'\{[^{}]+\}', text)
        if objects:
            try:
                items = [json.loads(obj) for obj in objects]
                return self._clean_items(items), None
            except json.JSONDecodeError:
                pass

        # Strategy 5: plain text fallback — extract food-like words line by line
        lines = [l.strip(" -•*\t") for l in raw_text.splitlines() if l.strip()]
        plain_items = []
        for line in lines:
            # Skip lines that look like JSON structure chars or very long lines
            if line in ("{", "}", "[", "]") or len(line) > 100:
                continue
            # Try to extract "name" value from partial JSON
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', line)
            if name_match:
                plain_items.append({"name": name_match.group(1), "quantity": "unknown", "confidence": "low"})
            elif re.match(r'^[a-z][a-z\s]{1,40}$', line.lower()):
                plain_items.append({"name": line.lower(), "quantity": "unknown", "confidence": "low"})

        if plain_items:
            logger.warning("Used plain-text fallback parsing — %d items found.", len(plain_items))
            return plain_items, None

        logger.error("All parsing strategies failed. Raw: %s", raw_text[:200])
        return [], "Could not parse API response. Try a clearer photo."

    def _clean_items(self, items: list) -> List[dict]:
        """Validate and clean parsed ingredient items."""
        cleaned = []
        seen_names = set()
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = str(item["name"]).strip().lower()
            # Skip blanks, very short (likely noise), or duplicates
            if not name or len(name) < 2 or name in seen_names:
                continue
            # Skip obviously non-food strings
            if any(skip in name for skip in ["json", "array", "format", "example"]):
                continue
            seen_names.add(name)
            cleaned.append({
                "name": name,
                "quantity": str(item.get("quantity", "unknown")).strip(),
                "confidence": str(item.get("confidence", "medium")).lower(),
            })
        return cleaned

    def detected_to_strings(self, detected: List[dict]) -> List[str]:
        """Convert detected ingredient dicts to human-readable strings."""
        result = []
        for item in detected:
            name = item.get("name", "")
            qty = item.get("quantity", "unknown")
            if qty and qty.lower() not in ("unknown", ""):
                result.append(f"{qty} {name}")
            else:
                result.append(name)
        return result

    def scan_barcode_or_label(
        self, image_bytes: bytes, mime_type: str = "image/jpeg"
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Scan a packaged food label or barcode image and extract nutrition data.

        Returns (nutrition_dict, error_message).
        nutrition_dict keys:
            product_name, brand, serving_size, servings_per_container,
            calories_per_serving, protein_g, carbs_g, sugar_g, fat_g,
            saturated_fat_g, sodium_mg, fiber_g, ingredient_list, allergens, confidence
        """
        if not self._available:
            return None, "Vision API not configured. Add a Groq or Gemini API key first."

        try:
            if self._provider == "openai":
                b64 = base64.b64encode(image_bytes).decode()
                response = self._client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                            {"type": "text", "text": _BARCODE_PROMPT},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw_text = response.choices[0].message.content.strip()
            elif self._provider == "groq":
                raw_list, err = self._detect_groq(image_bytes, mime_type, _BARCODE_PROMPT)
                # Groq returns list form — need to call raw for object
                # Re-call with object extraction
                b64 = base64.b64encode(image_bytes).decode()
                response = self._client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                            {"type": "text", "text": _BARCODE_PROMPT},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=1024,
                )
                raw_text = response.choices[0].message.content.strip()
            else:
                # Gemini
                from google.genai import types
                image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                for model in ["gemini-2.5-flash-lite", "gemini-1.5-flash", "gemini-2.0-flash"]:
                    try:
                        response = self._client.models.generate_content(
                            model=model,
                            contents=[_BARCODE_PROMPT, image_part],
                            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1024),
                        )
                        raw_text = response.text.strip()
                        break
                    except Exception as e:
                        if "429" in str(e):
                            import time
                            time.sleep(10)
                        continue
                else:
                    return None, "All Gemini models exhausted."

            # Parse the JSON object
            raw_text = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
            raw_text = re.sub(r"```\s*$", "", raw_text).strip()
            # Extract JSON object
            obj_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if obj_match:
                data = json.loads(obj_match.group(0))
                return data, None
            return None, "Could not parse label data. Try a clearer photo."

        except Exception as e:
            logger.error("scan_barcode_or_label error: %s", e)
            return None, f"Scan error: {e}"

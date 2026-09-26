"""
Stage 3: LLM Classification Node (Single Prompt Call)

Takes the cleaned cargo text + Top-K candidate HS codes from retrieval
and makes ONE LLM call to:
1. Select the best HS code from the candidates
2. Generate a full Chapter → Heading → Subheading reasoning chain
3. Output a calibrated confidence score
4. For ambiguous cases, explain trade-offs between top candidates

Uses Google Gemini (gemini-3.8-flash) via the new google-genai SDK.
"""

import json
import re
from google import genai
from google.genai import types

from app.config import GOOGLE_API_KEY, LLM_MODEL, CONFIDENCE_THRESHOLD
from app.schemas import (
    HSCandidate, ClassificationResult, ClassificationReasoning,
)


# Initialize the new Genai client
_client = genai.Client(api_key=GOOGLE_API_KEY)

CLASSIFICATION_PROMPT = """You are TariffAI, an expert customs tariff classification specialist.
You classify cargo descriptions into 6-digit Harmonized System (HS) codes.

## Your Task
Given a cargo description and a set of candidate HS codes retrieved from the tariff schedule,
select the single most accurate 6-digit HS code and explain your reasoning.

## Cargo Description (cleaned)
{cleaned_text}

## Candidate HS Codes
{candidates_text}

## Instructions
1. Analyze the cargo description to identify key product attributes:
   - Material/composition (cotton, polyester, steel, etc.)
   - Product type (shirt, rice, tile, etc.)
   - Processing state (raw, milled, knitted, woven, etc.)
   - End use (apparel, construction, food, etc.)

2. Walk through the HS hierarchy top-down:
   - SECTION: Which broad category does this product belong to?
   - CHAPTER (2-digit): Which chapter matches the product type?
   - HEADING (4-digit): Which heading captures the specific product?
   - SUBHEADING (6-digit): Which subheading matches the detailed attributes?

3. If the classification is ambiguous between candidates, explain the trade-offs.

4. Assign a confidence score:
   - 0.9-1.0: Clear, unambiguous match
   - 0.7-0.89: Strong match but minor ambiguity possible
   - 0.5-0.69: Ambiguous — multiple valid interpretations
   - Below 0.5: Insufficient information to classify reliably

## Required Output Format (JSON)
{{
    "primary_code": "620462",
    "primary_description": "The HS description for the selected code",
    "confidence": 0.85,
    "reasoning": {{
        "section_reasoning": "Why this section...",
        "chapter_reasoning": "Why this chapter...",
        "heading_reasoning": "Why this heading...",
        "subheading_reasoning": "Why this specific subheading...",
        "key_factors": ["factor1", "factor2"]
    }},
    "alternative_codes": [
        {{
            "code": "620463",
            "description": "Alternative description",
            "why_not": "Why this was not selected",
            "trade_off": "What would make this the correct choice instead"
        }}
    ]
}}

Respond with ONLY the JSON object. No markdown fences, no preamble."""


def _format_candidates(candidates: list[HSCandidate]) -> str:
    """Format candidates into a readable text block for the prompt."""
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. Code: {c.code}\n"
            f"   Description: {c.description}\n"
            f"   Heading ({c.heading_code}): {c.heading_description}\n"
            f"   Chapter ({c.chapter_code}): {c.chapter_description}\n"
            f"   Section {c.section}: {c.section_description}\n"
            f"   Retrieval Score: {c.retrieval_score} ({c.retrieval_method})"
        )
    return "\n\n".join(lines)


def classify_with_llm(
    cleaned_text: str,
    candidates: list[HSCandidate],
    raw_text: str = "",
) -> ClassificationResult:
    """
    Make a single LLM call to classify the cargo description.

    Args:
        cleaned_text: Preprocessed cargo text (from Stage 1)
        candidates: Top-K candidate HS codes (from Stage 2)
        raw_text: Original raw text (for the output record)

    Returns:
        ClassificationResult with code, reasoning chain, and confidence
    """
    candidates_text = _format_candidates(candidates)

    prompt = CLASSIFICATION_PROMPT.format(
        cleaned_text=cleaned_text,
        candidates_text=candidates_text,
    )

    response = _client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,  # Low temperature for deterministic classification
        ),
    )

    response_text = response.text.strip()

    # Parse the JSON response
    try:
        result_data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences if present
        json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group(1).strip())
        else:
            raise ValueError(f"LLM returned invalid JSON: {response_text[:200]}")

    # Build the reasoning chain
    reasoning_data = result_data.get("reasoning", {})
    reasoning = ClassificationReasoning(
        section_reasoning=reasoning_data.get("section_reasoning", ""),
        chapter_reasoning=reasoning_data.get("chapter_reasoning", ""),
        heading_reasoning=reasoning_data.get("heading_reasoning", ""),
        subheading_reasoning=reasoning_data.get("subheading_reasoning", ""),
        key_factors=reasoning_data.get("key_factors", []),
    )

    # Find the matching candidate for hierarchy info
    primary_code = result_data.get("primary_code") or None
    primary_desc = result_data.get("primary_description") or ""
    hierarchy = {}
    if primary_code:
        for c in candidates:
            if c.code == primary_code:
                hierarchy = {
                    "section": {"code": c.section, "description": c.section_description},
                    "chapter": {"code": c.chapter_code, "description": c.chapter_description},
                    "heading": {"code": c.heading_code, "description": c.heading_description},
                    "subheading": {"code": c.code, "description": c.description},
                }
                break

    confidence = float(result_data.get("confidence", 0.5))
    is_ambiguous = confidence < CONFIDENCE_THRESHOLD or primary_code is None

    return ClassificationResult(
        input_text=raw_text or cleaned_text,
        cleaned_text=cleaned_text,
        primary_code=primary_code,
        primary_description=primary_desc,
        confidence=confidence,
        reasoning=reasoning,
        hierarchy=hierarchy,
        alternative_codes=result_data.get("alternative_codes", []),
        is_ambiguous=is_ambiguous,
    )

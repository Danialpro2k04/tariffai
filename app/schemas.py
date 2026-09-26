from pydantic import BaseModel, Field
from typing import Literal


class HSCodeEntry(BaseModel):
    """A single HS code entry from the tariff schedule."""
    code: str = Field(..., description="6-digit HS code, e.g. '620462'")
    description: str = Field(..., description="Official HS description text")
    chapter_code: str = Field(..., description="2-digit chapter code, e.g. '62'")
    chapter_description: str = Field(..., description="Chapter-level description")
    heading_code: str = Field(..., description="4-digit heading code, e.g. '6204'")
    heading_description: str = Field(..., description="Heading-level description")
    section: str = Field(default="", description="Section number (Roman numeral)")
    section_description: str = Field(default="", description="Section-level description")


class HSCandidate(BaseModel):
    """A candidate HS code returned by the retrieval stage."""
    code: str
    description: str
    chapter_code: str
    chapter_description: str
    heading_code: str
    heading_description: str
    section: str = ""
    section_description: str = ""
    retrieval_score: float = Field(..., description="Combined retrieval score (0-1)")
    retrieval_method: str = Field(..., description="How this candidate was found: 'semantic', 'bm25', or 'hybrid'")


class ClassificationReasoning(BaseModel):
    """Step-by-step reasoning chain for the classification decision."""
    section_reasoning: str = Field(..., description="Why this section was chosen")
    chapter_reasoning: str = Field(..., description="Why this chapter was chosen")
    heading_reasoning: str = Field(..., description="Why this heading was chosen")
    subheading_reasoning: str = Field(..., description="Why this subheading was chosen")
    key_factors: list[str] = Field(default_factory=list, description="Key product attributes that drove the decision")


class ClassificationResult(BaseModel):
    """Final classification output for a single cargo description."""
    input_text: str = Field(..., description="Original raw cargo description")
    cleaned_text: str = Field(..., description="Preprocessed cargo text")
    primary_code: str | None = Field(default=None, description="Recommended 6-digit HS code")
    primary_description: str | None = Field(default="", description="HS description for the primary code")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence (0-1)")
    reasoning: ClassificationReasoning = Field(..., description="Step-by-step reasoning chain")
    hierarchy: dict = Field(
        default_factory=dict,
        description="Full hierarchy: {section, chapter, heading, subheading} with codes and descriptions"
    )
    alternative_codes: list[dict] = Field(
        default_factory=list,
        description="Top alternative candidates with trade-offs (for ambiguous cases)"
    )
    is_ambiguous: bool = Field(default=False, description="True if confidence < threshold")
    processing_time_ms: float = Field(default=0.0, description="Total processing time in milliseconds")


class ClassifyRequest(BaseModel):
    """API request to classify a cargo description."""
    description: str = Field(
        ...,
        min_length=3,
        description="Natural language cargo description from B/L or invoice",
        examples=[
            "500 cartons of men's cotton polo shirts, 100% cotton, knitted, sizes M-XXL",
            "1200 bags of basmati rice, long grain, milled, 25kg bags",
            "20 pallets of ceramic floor tiles 30x30cm glazed",
        ]
    )


class BatchClassifyRequest(BaseModel):
    """API request to classify multiple cargo descriptions."""
    descriptions: list[str] = Field(..., min_length=1, max_length=50)


class ClassifyResponse(BaseModel):
    """API response for a classification request."""
    status: Literal["classified", "ambiguous", "error"]
    result: ClassificationResult | None = None
    error: str | None = None

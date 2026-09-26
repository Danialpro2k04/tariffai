"""
LangGraph Classification Pipeline

Orchestrates the 3-stage TariffAI pipeline as a LangGraph state machine:

    START → preprocess → retrieve_candidates → classify → END
                                                  │
                                         (if confidence < 0.7)
                                                  │
                                                  ▼
                                          ambiguous_output
                                        (top-3 with trade-offs)

The state machine provides:
- Clear node boundaries for debugging and logging
- Checkpointing support for auditing (prepare for Week 6)
- Clean separation of concerns (no LLM in stages 1-2)
"""

import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END

from app.cargo_preprocessor import preprocess_cargo_text
from app.hs_retriever import retrieve_candidates
from app.llm_classifier import classify_with_llm
from app.schemas import HSCandidate, ClassificationResult
from app.config import CONFIDENCE_THRESHOLD


# ── LangGraph State ─────────────────────────────────────────────────────

class ClassificationState(TypedDict):
    """State that flows through the classification pipeline."""
    # Inputs
    raw_text: str

    # Stage 1 output
    cleaned_text: str

    # Stage 2 output
    candidates: list[dict]  # Serialized HSCandidate dicts (LangGraph needs serializable state)

    # Stage 3 output
    result: dict | None  # Serialized ClassificationResult

    # Metadata
    start_time: float
    processing_time_ms: float


# ── Node Functions ──────────────────────────────────────────────────────

def preprocess_node(state: ClassificationState) -> dict:
    """
    Stage 1: Clean the raw cargo text.
    Pure Python/regex — no LLM, no network calls, sub-millisecond.
    """
    raw = state["raw_text"]
    cleaned = preprocess_cargo_text(raw)
    return {
        "cleaned_text": cleaned,
        "start_time": time.time(),
    }


def retrieve_node(state: ClassificationState) -> dict:
    """
    Stage 2: Retrieve Top-K candidate HS codes.
    Qdrant semantic search + BM25 keyword search, fused via RRF.
    No LLM involved.
    """
    cleaned = state["cleaned_text"]
    candidates = retrieve_candidates(cleaned)

    # Serialize candidates for LangGraph state
    candidates_dicts = [c.model_dump() for c in candidates]
    return {"candidates": candidates_dicts}


def classify_node(state: ClassificationState) -> dict:
    """
    Stage 3: LLM classification.
    Single Gemini call with cleaned text + candidates → final code + reasoning.
    """
    cleaned = state["cleaned_text"]
    raw = state["raw_text"]

    # Deserialize candidates
    candidates = [HSCandidate(**c) for c in state["candidates"]]

    # Classify
    result = classify_with_llm(cleaned, candidates, raw_text=raw)

    # Calculate total processing time
    elapsed = (time.time() - state.get("start_time", time.time())) * 1000
    result.processing_time_ms = elapsed

    return {"result": result.model_dump(), "processing_time_ms": elapsed}


# ── Conditional Edge ────────────────────────────────────────────────────

def check_ambiguity(state: ClassificationState) -> str:
    """
    Route based on confidence: if ambiguous, go to the ambiguous output
    node that formats the top-3 with trade-offs.
    """
    result = state.get("result")
    if result and result.get("is_ambiguous", False):
        return "ambiguous"
    return "done"


def ambiguous_output_node(state: ClassificationState) -> dict:
    """
    For ambiguous cases (confidence < threshold), ensure the result
    prominently surfaces the alternative codes with trade-off explanations.
    The result already contains alternatives from the LLM — this node
    is a hook for future enhancements (e.g., triggering human review).
    """
    # In the current implementation, the LLM already provides alternatives.
    # This node exists for:
    # 1. Future Week 6 integration: queue for human-in-the-loop review
    # 2. Logging ambiguous cases for model improvement
    # 3. Potential second LLM call with stricter disambiguation prompt
    return {}


# ── Build the Graph ─────────────────────────────────────────────────────

def build_classification_graph() -> StateGraph:
    """
    Construct the LangGraph state machine for HS code classification.
    """
    graph = StateGraph(ClassificationState)

    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("classify", classify_node)
    graph.add_node("ambiguous_output", ambiguous_output_node)

    # Wire edges
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "retrieve")
    graph.add_edge("retrieve", "classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        check_ambiguity,
        {
            "done": END,
            "ambiguous": "ambiguous_output",
        }
    )
    graph.add_edge("ambiguous_output", END)

    return graph


# ── Compiled Graph (singleton) ──────────────────────────────────────────

_compiled_graph = None


def get_classification_graph():
    """Get or create the compiled classification graph."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_classification_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


def classify_cargo(raw_text: str) -> ClassificationResult:
    """
    Main entry point: classify a raw cargo description.

    This is the function the FastAPI endpoint calls.
    """
    graph = get_classification_graph()

    initial_state: ClassificationState = {
        "raw_text": raw_text,
        "cleaned_text": "",
        "candidates": [],
        "result": None,
        "start_time": time.time(),
        "processing_time_ms": 0.0,
    }

    # Run the graph
    final_state = graph.invoke(initial_state)

    # Deserialize the result
    result_dict = final_state.get("result")
    if result_dict is None:
        raise RuntimeError("Classification pipeline produced no result")

    return ClassificationResult(**result_dict)

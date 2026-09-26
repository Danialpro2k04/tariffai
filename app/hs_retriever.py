"""
Stage 2: Hybrid HS Code Retriever (No LLM)

Combines two retrieval strategies to find the Top-K candidate HS codes:

1. **Semantic Search (Qdrant)**: Dense vector similarity using the same
   multilingual-e5-small model as SanctionScope. Captures meaning even when
   the cargo text uses different words than the official HS description
   (e.g., "polo shirts" → "men's shirts, knitted").

2. **BM25 Keyword Search**: Sparse term-frequency matching. Catches exact
   domain terms that embeddings might blur (e.g., "basmati" vs "rice" —
   both are semantically "rice" but BM25 will boost the basmati-specific entry).

The two result sets are fused using Reciprocal Rank Fusion (RRF) to produce
a single ranked candidate list.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from app.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL, TOP_K_CANDIDATES, SEMANTIC_WEIGHT, BM25_WEIGHT,
)
from app.schemas import HSCandidate
from app.cargo_preprocessor import extract_product_keywords


# ── Lazy-loaded singletons ──────────────────────────────────────────────
_qdrant_client: QdrantClient | None = None
_embed_model: SentenceTransformer | None = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus_map: list[dict] | None = None  # maps BM25 doc index → HS entry


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30.0, check_compatibility=False)
    return _qdrant_client


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Build a BM25 index from the HS codes stored in Qdrant.
    We scroll through all points and tokenize their descriptions.
    The index is built once at startup and cached.
    """
    global _bm25_index, _bm25_corpus_map

    if _bm25_index is not None and _bm25_corpus_map is not None:
        return _bm25_index, _bm25_corpus_map

    client = _get_qdrant()
    corpus_texts: list[list[str]] = []
    corpus_map: list[dict] = []

    # Scroll through all HS code entries in Qdrant
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            payload = point.payload
            # Build a combined text for BM25 from description + heading + chapter
            combined = " ".join([
                payload.get("description", ""),
                payload.get("heading_description", ""),
                payload.get("chapter_description", ""),
            ]).lower()
            tokens = combined.split()
            corpus_texts.append(tokens)
            corpus_map.append(payload)

        if next_offset is None:
            break
        offset = next_offset

    _bm25_index = BM25Okapi(corpus_texts)
    _bm25_corpus_map = corpus_map
    return _bm25_index, _bm25_corpus_map


# ── Semantic Search ─────────────────────────────────────────────────────

def _search_semantic(query_text: str, top_k: int) -> list[tuple[dict, float]]:
    """
    Embed the query and search Qdrant for nearest HS code vectors.
    Returns list of (payload_dict, score) tuples.
    """
    model = _get_embed_model()
    client = _get_qdrant()

    # multilingual-e5 expects "query: " prefix for retrieval queries
    query_vec = model.encode(f"query: {query_text}").tolist()

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vec,
            limit=top_k * 2,
            with_payload=True,
        )
        return [(hit.payload, hit.score) for hit in response.points]
    else:
        results = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vec,
            limit=top_k * 2,  # over-fetch for fusion
            with_payload=True,
        )
        return [(hit.payload, hit.score) for hit in results]


# ── BM25 Search ─────────────────────────────────────────────────────────

def _search_bm25(query_text: str, top_k: int) -> list[tuple[dict, float]]:
    """
    BM25 keyword search over HS code descriptions.
    Returns list of (payload_dict, score) tuples.
    """
    bm25, corpus_map = _build_bm25_index()
    tokens = query_text.lower().split()
    scores = bm25.get_scores(tokens)

    # Get top indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k * 2]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append((corpus_map[idx], float(scores[idx])))

    return results


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────

def _reciprocal_rank_fusion(
    semantic_results: list[tuple[dict, float]],
    bm25_results: list[tuple[dict, float]],
    k: int = 60,
) -> list[tuple[dict, float, str]]:
    """
    Fuse semantic and BM25 results using Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank_i) across all result lists.
    This is more robust than raw score normalization because it doesn't
    depend on score distributions being comparable.

    Returns: list of (payload, rrf_score, method) tuples, sorted by score.
    """
    code_scores: dict[str, float] = {}
    code_payloads: dict[str, dict] = {}
    code_methods: dict[str, set] = {}

    # Score semantic results
    for rank, (payload, score) in enumerate(semantic_results):
        code = payload.get("code", "")
        rrf = SEMANTIC_WEIGHT * (1.0 / (k + rank + 1))
        code_scores[code] = code_scores.get(code, 0.0) + rrf
        code_payloads[code] = payload
        code_methods.setdefault(code, set()).add("semantic")

    # Score BM25 results
    for rank, (payload, score) in enumerate(bm25_results):
        code = payload.get("code", "")
        rrf = BM25_WEIGHT * (1.0 / (k + rank + 1))
        code_scores[code] = code_scores.get(code, 0.0) + rrf
        code_payloads[code] = payload
        code_methods.setdefault(code, set()).add("bm25")

    # Sort by fused score
    sorted_codes = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for code, score in sorted_codes:
        methods = code_methods.get(code, set())
        method_str = "hybrid" if len(methods) > 1 else methods.pop()
        results.append((code_payloads[code], score, method_str))

    return results


# ── Public API ──────────────────────────────────────────────────────────

def retrieve_candidates(cleaned_text: str, top_k: int | None = None) -> list[HSCandidate]:
    """
    Retrieve Top-K candidate HS codes using hybrid semantic + BM25 search.

    Args:
        cleaned_text: Preprocessed cargo description (output of Stage 1)
        top_k: Number of candidates to return (default: from config)

    Returns:
        List of HSCandidate objects ranked by retrieval score
    """
    if top_k is None:
        top_k = TOP_K_CANDIDATES

    # Run both retrieval strategies
    semantic_results = _search_semantic(cleaned_text, top_k)
    bm25_results = _search_bm25(cleaned_text, top_k)

    # Fuse results
    fused = _reciprocal_rank_fusion(semantic_results, bm25_results)

    # Build candidate objects (take top_k)
    candidates = []
    for payload, score, method in fused[:top_k]:
        # Normalize the RRF score to 0-1 range for cleaner output
        # Max possible RRF score ≈ 1/(k+1) * (semantic_weight + bm25_weight)
        max_possible = (SEMANTIC_WEIGHT + BM25_WEIGHT) / 61  # k=60, rank=0
        normalized_score = min(score / max_possible, 1.0) if max_possible > 0 else 0.0

        candidates.append(HSCandidate(
            code=payload.get("code", ""),
            description=payload.get("description", ""),
            chapter_code=payload.get("chapter_code", ""),
            chapter_description=payload.get("chapter_description", ""),
            heading_code=payload.get("heading_code", ""),
            heading_description=payload.get("heading_description", ""),
            section=payload.get("section", ""),
            section_description=payload.get("section_description", ""),
            retrieval_score=round(normalized_score, 4),
            retrieval_method=method,
        ))

    return candidates


def initialize_retriever():
    """
    Pre-warm the retriever: load the embedding model and build the BM25 index.
    Call this at application startup so the first request isn't slow.
    """
    _get_embed_model()
    _build_bm25_index()
    print(f"[TariffAI] Retriever initialized: embedding model loaded, BM25 index built")

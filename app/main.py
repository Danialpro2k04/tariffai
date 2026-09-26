"""
TariffAI — FastAPI Application

Endpoints:
    POST /classify         — Classify a single cargo description
    POST /classify/batch   — Classify multiple descriptions
    GET  /health           — Health check with Qdrant status
"""

import time
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ClassifyRequest, BatchClassifyRequest,
    ClassifyResponse, ClassificationResult,
)
from app.graph import classify_cargo
from app.hs_retriever import initialize_retriever
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the retriever on startup."""
    print("[TariffAI] Starting up...")
    try:
        initialize_retriever()
        print("[TariffAI] ✅ Ready to classify!")
    except Exception as e:
        print(f"[TariffAI] ⚠️  Retriever init failed (Qdrant may not be running): {e}")
        print("[TariffAI]    The API will start but /classify will fail until Qdrant is available.")
    yield
    print("[TariffAI] Shutting down.")


app = FastAPI(
    title="TariffAI",
    description=(
        "Intelligent HS Code Classification Engine for Trade Compliance. "
        "Part of the TradeGuard platform (Week 4)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(req: ClassifyRequest):
    """
    Classify a cargo description into a 6-digit HS code.

    The pipeline:
    1. Preprocesses the raw text (regex, no LLM)
    2. Retrieves top candidates via hybrid search (Qdrant + BM25, no LLM)
    3. Makes ONE LLM call to select the best code with reasoning

    Returns the classification result including:
    - The recommended HS code
    - Full Chapter → Heading → Subheading reasoning chain
    - Confidence score
    - Alternative codes (for ambiguous cases)
    """
    try:
        result = classify_cargo(req.description)
        status = "ambiguous" if result.is_ambiguous else "classified"
        return ClassifyResponse(status=status, result=result)
    except Exception as e:
        traceback.print_exc()
        return ClassifyResponse(status="error", error=str(e))


@app.post("/classify/batch", response_model=list[ClassifyResponse])
def batch_classify_endpoint(req: BatchClassifyRequest):
    """
    Classify multiple cargo descriptions in one request.
    Each description is processed independently through the full pipeline.
    """
    results = []
    for desc in req.descriptions:
        try:
            result = classify_cargo(desc)
            status = "ambiguous" if result.is_ambiguous else "classified"
            results.append(ClassifyResponse(status=status, result=result))
        except Exception as e:
            results.append(ClassifyResponse(status="error", error=str(e)))
    return results


@app.get("/health")
def health_check():
    """Health check with Qdrant connectivity status."""
    from qdrant_client import QdrantClient

    qdrant_ok = False
    collection_count = 0
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0, check_compatibility=False)
        info = client.get_collection(QDRANT_COLLECTION)
        qdrant_ok = True
        collection_count = info.points_count
    except Exception:
        pass

    return {
        "service": "TariffAI",
        "status": "healthy" if qdrant_ok else "degraded",
        "qdrant": {
            "connected": qdrant_ok,
            "collection": QDRANT_COLLECTION,
            "points_count": collection_count,
        },
    }

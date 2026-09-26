# 🎯 TariffAI — Intelligent HS Code Classification Engine

> **Week 4 of the [TradeGuard](https://github.com/your-username/tradeguard) Platform**
> An AI-powered engine that classifies messy cargo descriptions from Bills of Lading into accurate 6-digit Harmonized System (HS) codes — with full reasoning chains and confidence scoring.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Pipeline-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?logo=qdrant)
![Gemini](https://img.shields.io/badge/Google_Gemini-2.0_Flash-4285F4?logo=google&logoColor=white)

---

## 🧠 The Problem

Standard RAG fails for HS code classification because:

- **Chunking destroys hierarchy** — HS codes are a tree structure (Section → Chapter → Heading → Subheading). Chunking a tariff schedule into 512-token blocks breaks this parent-child relationship.
- **Cargo descriptions are noisy** — B/L text contains container IDs (`MSCU1234567`), weights (`NET WT: 5000 KGS`), seal numbers, PO references, and boilerplate (`SAID TO CONTAIN`).
- **Embedding similarity alone can't distinguish** between `6204.62` (women's cotton trousers) and `6204.63` (women's synthetic trousers) — the descriptions are semantically near-identical.

## 💡 The Solution: 3-Stage Pipeline (Not Standard RAG)

Instead of naive chunk-and-embed, TariffAI uses a **staged retrieval + reasoning pipeline** where the LLM is only called once, at the very end:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  [Messy B/L Cargo Text]                                         │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────┐                           │
│  │ Stage 1: Regex Preprocessor     │  ◄── No LLM, ~0.1ms       │
│  │ Strips container IDs, weights,  │                            │
│  │ seals, PO#s, boilerplate        │                            │
│  └──────────────┬───────────────────┘                           │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                           │
│  │ Stage 2: Hybrid Retrieval       │  ◄── No LLM, ~50ms        │
│  │ Qdrant Semantic + BM25 Keyword  │                            │
│  │ Fused via Reciprocal Rank       │                            │
│  │ Fusion (RRF) → Top-5 Candidates │                            │
│  └──────────────┬───────────────────┘                           │
│                 │                                                │
│                 ▼                                                │
│  ┌──────────────────────────────────┐                           │
│  │ Stage 3: LLM Classification     │  ◄── 1 Gemini call, ~1s   │
│  │ Selects best code from Top-5    │                            │
│  │ Generates reasoning chain:      │                            │
│  │ Section → Chapter → Heading     │                            │
│  │ → Subheading                    │                            │
│  │ + Confidence Score              │                            │
│  └──────────────────────────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Design Decision | Rationale |
|---|---|
| **Hybrid search (Semantic + BM25)** | BM25 catches exact domain terms like "basmati" that embeddings blur into generic "rice". Semantic search handles synonyms and rephrasings. |
| **Reciprocal Rank Fusion (RRF)** | Rank-based fusion avoids score distribution mismatch between Qdrant cosine scores and BM25 term-frequency scores |
| **Single LLM call at the end** | Cost efficiency: ~$0.001/classification. Stages 1–2 are deterministic, fast, and free |
| **Full hierarchy in embedding text** | Embedding "Section XI: Textiles. Chapter 62: Apparel. 620462: Of cotton" gives much richer semantic context than just "Of cotton" |
| **Regex preprocessing (not LLM)** | B/L noise patterns are finite and well-defined. Regex is deterministic, free, and sub-millisecond |

---

## 🏗️ LangGraph State Machine

TariffAI is orchestrated as a **LangGraph state machine** — providing clean node boundaries for debugging, logging, and future audit trail integration:

```
START → preprocess → retrieve → classify ──→ END
                                    │
                             (confidence < 0.7)
                                    │
                                    ▼
                          ambiguous_output ──→ END
                        (top-3 with trade-offs)
```

The `ambiguous_output` node is a hook for future **human-in-the-loop review** — when the classifier isn't confident enough, it surfaces the top-3 candidates with trade-off explanations so a customs broker can make the final call.

---

## 📦 Project Structure

```
tariffai/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Centralized env config
│   ├── schemas.py                # All Pydantic models (request/response/state)
│   ├── cargo_preprocessor.py     # Stage 1: Regex noise stripping (No LLM)
│   ├── hs_retriever.py           # Stage 2: Qdrant + BM25 hybrid search (No LLM)
│   ├── llm_classifier.py         # Stage 3: Gemini classification (1 LLM call)
│   ├── graph.py                  # LangGraph state machine orchestrator
│   └── main.py                   # FastAPI endpoints
├── scripts/
│   ├── __init__.py
│   └── ingest_hs_codes.py        # Downloads HS CSV → embeds → loads Qdrant
├── tests/
│   └── test_preprocessor.py      # 12 unit tests for Stage 1
├── docker-compose.yml            # Qdrant container
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for Qdrant)
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/tariffai.git
cd tariffai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Start Qdrant

```bash
docker compose up -d
```

### 4. Ingest HS Code Data

This downloads the full WCO Harmonized System nomenclature (~5,300+ 6-digit codes), resolves the hierarchy, embeds each entry, and loads into Qdrant:

```bash
python -m scripts.ingest_hs_codes
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --port 8004
```

### 6. Classify!

```bash
curl -X POST http://localhost:8004/classify \
  -H "Content-Type: application/json" \
  -d '{"description": "500 cartons of mens cotton polo shirts, 100% cotton, knitted, sizes M-XXL"}'
```

---

## 📡 API Reference

### `POST /classify`

Classify a single cargo description into a 6-digit HS code.

**Request:**
```json
{
  "description": "500 cartons of mens cotton polo shirts, 100% cotton, knitted, sizes M-XXL"
}
```

**Response:**
```json
{
  "status": "classified",
  "result": {
    "input_text": "500 cartons of mens cotton polo shirts...",
    "cleaned_text": "mens cotton polo shirts, 100% cotton, knitted, sizes M-XXL",
    "primary_code": "610510",
    "primary_description": "Men's or boys' shirts, of cotton, knitted or crocheted",
    "confidence": 0.88,
    "reasoning": {
      "section_reasoning": "Section XI covers textiles and textile articles...",
      "chapter_reasoning": "Chapter 61 covers articles of apparel, knitted or crocheted...",
      "heading_reasoning": "Heading 6105 covers men's or boys' shirts, knitted...",
      "subheading_reasoning": "Subheading 610510 specifies cotton material...",
      "key_factors": ["knitted construction", "100% cotton", "men's apparel", "shirts/polo"]
    },
    "hierarchy": {
      "section": {"code": "XI", "description": "Textiles and textile articles"},
      "chapter": {"code": "61", "description": "Articles of apparel, knitted or crocheted"},
      "heading": {"code": "6105", "description": "Men's or boys' shirts, knitted"},
      "subheading": {"code": "610510", "description": "Of cotton"}
    },
    "alternative_codes": [
      {
        "code": "610520",
        "description": "Of man-made fibres",
        "why_not": "Product is specified as 100% cotton, not synthetic",
        "trade_off": "Would apply if the polo shirts contained synthetic blends"
      }
    ],
    "is_ambiguous": false,
    "processing_time_ms": 1243.5
  }
}
```

### `POST /classify/batch`

Classify multiple descriptions in one request (max 50).

```json
{
  "descriptions": [
    "basmati rice long grain milled 25kg bags",
    "ceramic floor tiles 30x30cm glazed",
    "frozen headless shrimp 500g packs"
  ]
}
```

### `GET /health`

Returns service health with Qdrant connectivity and collection status.

---

## 🔬 How It Works — Deep Dive

### Stage 1: Cargo Preprocessor (`cargo_preprocessor.py`)

Pure Python/regex. Strips 12+ categories of B/L noise:

| Noise Pattern | Example | Action |
|---|---|---|
| Container IDs | `MSCU1234567` | Removed |
| Seal numbers | `SEAL NO: ML12345` | Removed |
| PO / Reference numbers | `PO# 12345`, `REF: ABC-123` | Removed |
| Weights | `NET WT: 5000 KGS` | Removed |
| Volume | `25.5 CBM` | Removed |
| Dimensions | `120x80x100 CM` | Removed |
| Container types | `1X40HC`, `2X20GP` | Removed |
| Temperature settings | `TEMP SET AT -18C` | Removed |
| Existing HS codes | `HS CODE: 620462` | Removed (classifier should determine fresh) |
| Shipping boilerplate | `SAID TO CONTAIN`, `SHIPPER'S LOAD AND COUNT` | Removed |
| Packing unit counts | `500 CARTONS OF` | Removed (keeps product description) |

### Stage 2: Hybrid Retriever (`hs_retriever.py`)

Combines two retrieval strategies:

1. **Qdrant Semantic Search** — Dense vector similarity using `intfloat/multilingual-e5-small` (384-dim). Captures meaning even when wording differs from the official HS description.

2. **BM25 Keyword Search** — Sparse term-frequency matching. Catches exact domain terms that dense embeddings might blur.

Results are fused using **Reciprocal Rank Fusion (RRF)**:
```
RRF_score(code) = Σ  weight_i / (k + rank_i)
```
This is more robust than raw score normalization because it doesn't depend on score distributions being comparable.

### Stage 3: LLM Classifier (`llm_classifier.py`)

One structured Gemini call with:
- **Input:** Cleaned cargo text + Top-5 candidates with full hierarchy
- **Output:** JSON with selected code, reasoning chain, confidence, and alternatives
- **Temperature:** 0.1 (near-deterministic for consistent classification)
- **Response format:** `application/json` (Gemini structured output)

---

## 📊 HS Code Data Source

**[datasets/harmonized-system](https://github.com/datasets/harmonized-system)** — Frictionless Data project

- ~5,300+ entries: 21 Sections, 97 Chapters, 1,228 Headings, 5,000+ Subheadings
- Full parent-pointer hierarchy resolved during ingestion
- License: **ODC-PDDL** (Public Domain) — free for commercial use
- Auto-downloaded on first ingestion run

---

## 🧪 Testing

```bash
# Run Stage 1 unit tests (no external dependencies needed)
python tests/test_preprocessor.py
```

```
  ✅ test_strips_container_numbers
  ✅ test_strips_po_numbers
  ✅ test_strips_weight
  ✅ test_strips_seal_number
  ✅ test_strips_boilerplate
  ✅ test_strips_container_type
  ✅ test_strips_dimensions
  ✅ test_preserves_clean_description
  ✅ test_strips_existing_hs_code
  ✅ test_strips_multiple_noise_types
  ✅ test_extract_keywords
  ✅ test_empty_input

  Results: 12 passed, 0 failed
```

---

## 🛣️ Part of the TradeGuard Platform

TariffAI is **Week 4** of a 24-week roadmap building an enterprise-grade trade compliance platform:

```
Week 1-2: freight_document_scanner  →  B/L extraction + cross-validation
Week 3:   sanctionscope             →  Entity sanctions screening
Week 4:   tariffai                  →  HS code classification (this repo)
Week 5:   LangGraph Core Pipeline   →  Connects all modules into one agent
Week 6:   ComplianceVault           →  Audit trails, RBAC, human override
```

**Integration path:** The `freight_document_scanner` extracts cargo descriptions from raw B/L documents using Azure AI Document Intelligence. TariffAI takes that extracted text and classifies it. In Week 5, the LangGraph Core Pipeline will wire them together into an end-to-end automated flow.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST endpoints with auto-generated docs |
| Agent Pipeline | LangGraph | State machine orchestration with conditional routing |
| Vector Database | Qdrant | Semantic search over HS code embeddings |
| Embeddings | intfloat/multilingual-e5-small | 384-dim multilingual embeddings |
| Keyword Search | rank-bm25 | Sparse term-frequency matching |
| LLM | Google Gemini 2.0 Flash | Classification reasoning (1 call per request) |
| Data Validation | Pydantic v2 | Request/response schema validation |

---

## 📄 License

MIT

---

**Built by Danyal Wahdat** as part of the TradeGuard trade compliance platform.

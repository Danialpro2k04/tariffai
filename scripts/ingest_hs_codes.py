"""
HS Code Data Ingestion Pipeline

Downloads the public HS code dataset (from the Frictionless Data
`datasets/harmonized-system` CSV) and loads it into Qdrant with
full hierarchy metadata attached to each 6-digit subheading vector.

Data source: https://github.com/datasets/harmonized-system
License: ODC-PDDL (Public Domain)

Each Qdrant point represents a single 6-digit HS subheading and carries:
- vector: embedding of the combined description text
- payload: code, description, chapter_code, chapter_description,
           heading_code, heading_description, section, section_description
"""

import csv
import io
import sys
import os
import json
from pathlib import Path

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Add parent to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)


CSV_URL = "https://raw.githubusercontent.com/datasets/harmonized-system/master/data/harmonized-system.csv"
LOCAL_CACHE = Path(__file__).resolve().parent.parent / "data" / "harmonized-system.csv"
PROCESSED_CACHE = Path(__file__).resolve().parent.parent / "data" / "hs_processed.json"


def download_hs_data() -> str:
    """Download the HS CSV if not already cached locally."""
    LOCAL_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if LOCAL_CACHE.exists():
        print(f"[ingest] Using cached CSV: {LOCAL_CACHE}")
        return LOCAL_CACHE.read_text(encoding="utf-8")

    print(f"[ingest] Downloading HS data from {CSV_URL} ...")
    resp = requests.get(CSV_URL, timeout=60)
    resp.raise_for_status()
    LOCAL_CACHE.write_text(resp.text, encoding="utf-8")
    print(f"[ingest] Saved to {LOCAL_CACHE} ({len(resp.text)} bytes)")
    return resp.text


def parse_hierarchy(csv_text: str) -> list[dict]:
    """
    Parse the CSV into a flat list, then resolve the parent pointers
    to attach full hierarchy metadata to each 6-digit subheading.

    CSV columns: section, hscode, description, parent, level
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    # Build lookup: code/section → row
    lookup: dict[str, dict] = {}
    for row in rows:
        key = row["hscode"] if row["hscode"] else row["section"]
        lookup[key] = row

    # For each 6-digit code, walk up the parent chain
    entries = []
    for row in rows:
        code = row.get("hscode", "").strip()
        level = row.get("level", "").strip()
        description = row.get("description", "").strip()

        # We only index 6-digit subheadings (level=6)
        if level != "6" or not code or not description:
            continue

        # Walk up: 6-digit → 4-digit heading → 2-digit chapter → section
        heading_code = ""
        heading_desc = ""
        chapter_code = ""
        chapter_desc = ""
        section = ""
        section_desc = ""

        # Parent of 6-digit should be 4-digit heading
        parent_key = row.get("parent", "").strip()
        if parent_key and parent_key in lookup:
            heading_row = lookup[parent_key]
            heading_code = heading_row.get("hscode", "").strip()
            heading_desc = heading_row.get("description", "").strip()

            # Parent of 4-digit should be 2-digit chapter
            chapter_key = heading_row.get("parent", "").strip()
            if chapter_key and chapter_key in lookup:
                chapter_row = lookup[chapter_key]
                chapter_code = chapter_row.get("hscode", "").strip()
                chapter_desc = chapter_row.get("description", "").strip()

                # Parent of 2-digit should be section (Roman numeral)
                section_key = chapter_row.get("parent", "").strip()
                if section_key and section_key in lookup:
                    section_row = lookup[section_key]
                    section = section_row.get("section", section_key).strip()
                    section_desc = section_row.get("description", "").strip()

        entries.append({
            "code": code,
            "description": description,
            "heading_code": heading_code,
            "heading_description": heading_desc,
            "chapter_code": chapter_code,
            "chapter_description": chapter_desc,
            "section": section,
            "section_description": section_desc,
        })

    return entries


def build_embedding_text(entry: dict) -> str:
    """
    Build a rich text for embedding that captures the full hierarchy.
    This gives the embedding model maximum context for similarity matching.

    Example output:
        "Section XI: Textiles and textile articles.
         Chapter 62: Articles of apparel and clothing accessories, not knitted or crocheted.
         Heading 6204: Women's or girls' suits, ensembles, jackets, blazers, dresses, skirts, divided skirts, trousers.
         620462: Of cotton"
    """
    parts = []
    if entry["section"] and entry["section_description"]:
        parts.append(f"Section {entry['section']}: {entry['section_description']}")
    if entry["chapter_code"] and entry["chapter_description"]:
        parts.append(f"Chapter {entry['chapter_code']}: {entry['chapter_description']}")
    if entry["heading_code"] and entry["heading_description"]:
        parts.append(f"Heading {entry['heading_code']}: {entry['heading_description']}")
    parts.append(f"{entry['code']}: {entry['description']}")
    return ". ".join(parts)


def ingest_to_qdrant(entries: list[dict]):
    """Embed all HS entries and upsert into Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120.0)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Recreate collection
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[ingest] Created Qdrant collection '{QDRANT_COLLECTION}' (dim={EMBEDDING_DIM})")

    # Embed and upload in batches
    batch_size = 256
    total = len(entries)

    for i in range(0, total, batch_size):
        batch = entries[i:i + batch_size]

        # Build embedding texts with hierarchy context
        texts = [build_embedding_text(e) for e in batch]

        # multilingual-e5 expects "passage: " prefix for documents
        prefixed = [f"passage: {t}" for t in texts]
        vectors = model.encode(prefixed, show_progress_bar=False).tolist()

        points = []
        for j, (entry, vec) in enumerate(zip(batch, vectors)):
            points.append(PointStruct(
                id=i + j,
                vector=vec,
                payload=entry,
            ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        uploaded = min(i + batch_size, total)
        print(f"[ingest] Uploaded {uploaded}/{total} HS codes")

    print(f"\n[ingest] ✅ Successfully loaded {total} HS codes into Qdrant!")
    print(f"[ingest] Collection: {QDRANT_COLLECTION}")


def main():
    print("=" * 60)
    print("TariffAI — HS Code Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Download/load CSV
    csv_text = download_hs_data()

    # Step 2: Parse hierarchy
    entries = parse_hierarchy(csv_text)
    print(f"[ingest] Parsed {len(entries)} 6-digit HS subheadings")

    if not entries:
        print("[ingest] ERROR: No 6-digit codes found. Check CSV format.")
        sys.exit(1)

    # Save processed data for debugging / BM25 index
    PROCESSED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_CACHE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Saved processed entries to {PROCESSED_CACHE}")

    # Step 3: Embed and load into Qdrant
    ingest_to_qdrant(entries)

    # Print sample entries for verification
    print("\n[ingest] Sample entries:")
    for entry in entries[:3]:
        print(f"  {entry['code']}: {entry['description']}")
        print(f"    └─ Heading {entry['heading_code']}: {entry['heading_description'][:60]}...")
        print(f"    └─ Chapter {entry['chapter_code']}: {entry['chapter_description'][:60]}...")
        print()


if __name__ == "__main__":
    main()

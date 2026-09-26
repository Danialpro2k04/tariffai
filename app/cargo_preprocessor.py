"""
Stage 1: Cargo Description Preprocessor (No LLM)

Strips known B/L clutter from raw cargo text so downstream search
operates on clean product attributes. This is pure regex — fast,
deterministic, and free.

Common B/L noise that gets removed:
- Container numbers (e.g., MSCU1234567)
- PO / Reference numbers (PO#12345, REF: ABC-123)
- Weight / measurement strings (NET WT: 5000 KGS, 25 CBM)
- Shipping marks and seal numbers
- Dimension strings (120x80x100 CM)
- Packing counts that are not product descriptors (500 CTNS, 20 PLTS)
- Boilerplate phrases (SAID TO CONTAIN, SHIPPER'S LOAD AND COUNT)
"""

import re


# Pre-compiled patterns for performance
_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Container numbers: ABCU1234567
    ("container_id", re.compile(
        r"\b[A-Z]{4}\d{7}\b", re.IGNORECASE
    )),

    # Seal numbers: SEAL NO: 12345, SEAL# ML12345
    ("seal_number", re.compile(
        r"\bSEAL\s*(?:NO\.?|#)\s*:?\s*\S+", re.IGNORECASE
    )),

    # PO / Reference numbers: PO#12345, PO NO: 12345, REF: ABC-123, ORDER NO 456
    # The pattern requires a separator (space+NO/#/:/NUMBER) after the prefix keyword
    # to avoid false-positives on words like "polo", "invoice-ready", etc.
    ("po_ref", re.compile(
        r"\b(?:P\.?O\.?\s*(?:NO\.?|#|NUMBER)\s*:?\s*[\w\-/]+|"
        r"(?:REF|REFERENCE|ORDER|INV|INVOICE)\s*(?:NO\.?|#|NUMBER)?\s*:?\s*[\w\-/]+)",
        re.IGNORECASE
    )),

    # B/L number references
    ("bl_ref", re.compile(
        r"\b(?:B/?L|BILL\s+OF\s+LADING)\s*(?:NO\.?|#)?\s*:?\s*[\w\-/]+",
        re.IGNORECASE
    )),

    # Weight strings: NET WT 5000 KGS, GROSS WEIGHT: 12000 LBS, 500KG
    ("weight", re.compile(
        r"\b(?:NET|GROSS|TARE)?\s*(?:WT\.?|WEIGHT|WGT)\.?\s*:?\s*[\d,\.]+\s*(?:KGS?|LBS?|MTS?|TONS?)\b",
        re.IGNORECASE
    )),
    # Standalone weight: 5000 KGS, 12.5 MT
    ("weight_standalone", re.compile(
        r"\b[\d,\.]+\s*(?:KGS?|LBS?|MTS?|TONS?|METRIC\s+TONS?)\b",
        re.IGNORECASE
    )),

    # Volume / CBM: 25.5 CBM, 100 CUBIC METERS
    ("volume", re.compile(
        r"\b[\d,\.]+\s*(?:CBM|CUBIC\s+METERS?|CU\.?\s*M\.?)\b",
        re.IGNORECASE
    )),

    # Dimensions: 120x80x100 CM, 40' x 8' x 8'6"
    ("dimensions", re.compile(
        r"\b\d+[\s]*[xX×][\s]*\d+(?:[\s]*[xX×][\s]*\d+)?\s*(?:CM|MM|M|IN|FT|INCHES|FEET)?\b",
        re.IGNORECASE
    )),

    # Container type references: 1X40HC, 2X20GP, 40' CONTAINER
    ("container_type", re.compile(
        r"\b\d+\s*[xX×]\s*(?:20|40|45)\s*(?:GP|HC|HQ|OT|RF|FR|DC|STD|FT|DRY|REEFER)\b",
        re.IGNORECASE
    )),

    # Shipping marks block header
    ("marks_header", re.compile(
        r"\b(?:SHIPPING\s+)?MARKS?\s*(?:&|AND)?\s*(?:NOS?\.?|NUMBERS?)?\s*:?\s*",
        re.IGNORECASE
    )),

    # Temperature settings: TEMP SET AT -18C, TEMP SET -1.5C, TEMP -18C
    ("temperature", re.compile(
        r"\bTEMP(?:ERATURE)?\s*(?:SET\s*(?:AT|TO)?)?\s*:?\s*[-+]?\d+\.?\d*\s*°?\s*[CF]\b",
        re.IGNORECASE
    )),

    # Packing unit counts: 500 CTNS, 1200 BAGS OF, 80 WOODEN CASES, 25 WOODEN CRATES, 1500 MASTER CARTONS
    # Run before standalone packaging materials so counts + packaging are stripped as a unit
    ("packing_prefix", re.compile(
        r"\b\d+\s*(?:WOODEN|CORRUGATED|FIBER|CARDBOARD|STEEL|PLASTIC|JUTE|PP|POLY|PAPER|MASTER)?\s*(?:CTNS?|CARTONS?|BAGS?|BALES?|PKGS?|PACKAGES?|"
        r"PALLETS?|PLTS?|DRUMS?|ROLLS?|BUNDLES?|CASES?|BOXES?|PCS?|PIECES?|UNITS?|CRATES?|MODULES?)\s*(?:OF\s+)?",
        re.IGNORECASE
    )),

    # Packaging phrases and specifications:
    # PACKED IN 10KG CORRUGATED BOXES, IN WOODEN CASES, IN JUTE BAGS, ON WOODEN PALLETS
    ("packaging_packed_in", re.compile(
        r"\b(?:PACKED\s+IN|PACKED|PACKAGING\s+IN|INTO|IN)\s+(?:[\d,\.]+\s*(?:KGS?|LBS?)\s+)?(?:WOODEN|CORRUGATED|FIBER|CARDBOARD|STEEL|PLASTIC|JUTE|PP|POLY|PAPER|HEAVY\s+DUTY)?\s*(?:CARTONS?|BOXES?|BAGS?|CRATES?|CASES?|DRUMS?|BALES?|PALLETS?|CONTAINERS?)\b",
        re.IGNORECASE
    )),

    # Standalone packaging types and materials:
    # WOODEN CRATES, WOODEN CASES, JUTE BAGS, PP BAGS, FIBER DRUMS, CORRUGATED BOXES, MASTER CARTONS
    ("packaging_materials", re.compile(
        r"\b(?:WOODEN\s+(?:CASES?|CRATES?|PALLETS?|BOXES?|SKIDS?)|"
        r"JUTE\s+BAGS?|PP\s+BAGS?|FIBER\s+DRUMS?|CORRUGATED\s+BOXES?|"
        r"MASTER\s+CARTONS?|CARDBOARD\s+BOXES?|"
        r"PALLETIZED|SHRINK\s+WRAPPED|LOOSE\s+IN\s+BULK|IN\s+BULK)\b",
        re.IGNORECASE
    )),

    # Boilerplate phrases
    ("boilerplate", re.compile(
        r"\b(?:SAID\s+TO\s+CONTAIN|SHIPPER[''`]?S?\s+LOAD\s+AND\s+COUNT|"
        r"STC|S\.?L\.?C\.?|CLEAN\s+ON\s+BOARD|FREIGHT\s+(?:PRE)?PAID|"
        r"AS\s+PER\s+(?:ATTACHED|SHIPPING)\s+\w+|"
        r"LADEN\s+ON\s+BOARD|RECEIVED\s+FOR\s+SHIPMENT|"
        r"SHIPPED\s+ON\s+BOARD)\b",
        re.IGNORECASE
    )),

    # HS code references already in the text (we want the classifier to determine it fresh)
    ("existing_hs", re.compile(
        r"\b(?:HS|HTS|TARIFF)\s*(?:CODE|NO\.?|#)?\s*:?\s*\d{4,10}\b",
        re.IGNORECASE
    )),
]

# Extra whitespace cleanup
_MULTI_SPACE = re.compile(r"\s{2,}")
_LEADING_TRAILING = re.compile(r"^[\s,;.\-–—]+|[\s,;.\-–—]+$")


def preprocess_cargo_text(raw_text: str) -> str:
    """
    Clean a raw B/L cargo description by removing shipping noise.

    Args:
        raw_text: Raw cargo description from a Bill of Lading

    Returns:
        Cleaned text containing primarily product attributes
        (material, type, composition, style, use-case, etc.)
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Apply all stripping patterns
    for name, pattern in _PATTERNS:
        text = pattern.sub(" ", text)

    # Normalize whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _LEADING_TRAILING.sub("", text)

    # If we stripped everything, fall back to original (the input might just be
    # a clean product description with no B/L noise)
    if len(text.strip()) < 5:
        text = raw_text.strip()

    return text


def extract_product_keywords(cleaned_text: str) -> list[str]:
    """
    Extract salient product keywords for BM25 matching.
    Returns lowercased, deduplicated keyword tokens.
    """
    # Remove common non-descriptive words and packaging container terms
    stopwords = {
        "of", "the", "and", "or", "for", "with", "in", "on", "at", "to",
        "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "new", "used", "other", "various", "assorted", "mixed",
        "made", "containing", "consisting", "comprising",
        # Packaging terms that should not bias BM25 keyword matching
        "bags", "bag", "cartons", "carton", "boxes", "box", "cases", "case",
        "crates", "crate", "pallets", "pallet", "drums", "drum", "bales", "bale",
        "packages", "package", "pkgs", "pkg", "ctns", "ctn", "corrugated",
        "cardboard", "shrink", "wrapped", "packed", "packaging", "palletized",
    }

    tokens = re.findall(r"[a-zA-Z]{2,}", cleaned_text.lower())
    seen = set()
    keywords = []
    for t in tokens:
        if t not in stopwords and t not in seen:
            seen.add(t)
            keywords.append(t)
    return keywords

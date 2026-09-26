"""
Tests for the cargo preprocessor (Stage 1).
These run without any external dependencies — pure regex tests.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cargo_preprocessor import preprocess_cargo_text, extract_product_keywords


def test_strips_container_numbers():
    raw = "MSCU1234567 500 CARTONS OF MEN'S COTTON POLO SHIRTS"
    cleaned = preprocess_cargo_text(raw)
    assert "MSCU1234567" not in cleaned
    assert "COTTON" in cleaned.upper() or "cotton" in cleaned.lower()


def test_strips_po_numbers():
    raw = "PO# 12345 BASMATI RICE LONG GRAIN MILLED"
    cleaned = preprocess_cargo_text(raw)
    assert "PO" not in cleaned.upper() or "POLO" in cleaned.upper()
    assert "BASMATI" in cleaned.upper()


def test_strips_weight():
    raw = "CERAMIC FLOOR TILES 30X30CM NET WT: 5000 KGS"
    cleaned = preprocess_cargo_text(raw)
    assert "5000" not in cleaned
    assert "KGS" not in cleaned.upper()
    assert "CERAMIC" in cleaned.upper()


def test_strips_seal_number():
    raw = "SEAL NO: ML12345 FROZEN SHRIMP HEADLESS"
    cleaned = preprocess_cargo_text(raw)
    assert "SEAL" not in cleaned.upper()
    assert "SHRIMP" in cleaned.upper()


def test_strips_boilerplate():
    raw = "SAID TO CONTAIN 100 COTTON T-SHIRTS BLUE"
    cleaned = preprocess_cargo_text(raw)
    assert "SAID TO CONTAIN" not in cleaned.upper()
    assert "COTTON" in cleaned.upper()


def test_strips_container_type():
    raw = "1X40HC MEN'S WOVEN COTTON TROUSERS ASSORTED SIZES"
    cleaned = preprocess_cargo_text(raw)
    assert "1X40HC" not in cleaned.upper()
    assert "TROUSERS" in cleaned.upper()


def test_strips_dimensions():
    raw = "STEEL PIPES SEAMLESS 120x80x100 CM GRADE A"
    cleaned = preprocess_cargo_text(raw)
    assert "120x80x100" not in cleaned
    assert "STEEL" in cleaned.upper()


def test_preserves_clean_description():
    raw = "Men's cotton polo shirts, 100% cotton, knitted, sizes M-XXL"
    cleaned = preprocess_cargo_text(raw)
    # Clean descriptions should pass through mostly intact
    assert "cotton" in cleaned.lower()
    assert "polo" in cleaned.lower()
    assert "knitted" in cleaned.lower()


def test_strips_existing_hs_code():
    raw = "HS CODE: 620462 WOMEN'S COTTON TROUSERS"
    cleaned = preprocess_cargo_text(raw)
    assert "HS CODE" not in cleaned.upper()
    assert "TROUSERS" in cleaned.upper()


def test_strips_multiple_noise_types():
    raw = (
        "MSCU9876543 SEAL NO: ABC123 PO# 99999 "
        "1X20GP 500 CARTONS OF BASMATI RICE LONG GRAIN MILLED 25KG BAGS "
        "NET WT 12500 KGS GROSS WEIGHT: 13000 KGS "
        "SAID TO CONTAIN SHIPPER'S LOAD AND COUNT"
    )
    cleaned = preprocess_cargo_text(raw)
    assert "MSCU9876543" not in cleaned
    assert "SEAL" not in cleaned.upper()
    assert "PO" not in cleaned.upper() or "POLO" in cleaned.upper()
    assert "12500" not in cleaned
    assert "SAID TO CONTAIN" not in cleaned.upper()
    assert "BASMATI" in cleaned.upper()
    assert "RICE" in cleaned.upper()


def test_extract_keywords():
    text = "men's cotton polo shirts knitted blue"
    keywords = extract_product_keywords(text)
    assert "cotton" in keywords
    assert "polo" in keywords
    assert "shirts" in keywords
    assert "knitted" in keywords
    # Stopwords should be excluded
    assert "of" not in keywords
    assert "the" not in keywords


def test_strips_packaging_descriptors():
    # Test stripping wooden cases, jute bags, crates, corrugated boxes, temperature without colon
    raw1 = "80 WOODEN CASES AUTOMOTIVE CAST IRON DISC BRAKE ROTORS"
    cleaned1 = preprocess_cargo_text(raw1)
    assert "WOODEN" not in cleaned1.upper()
    assert "CASES" not in cleaned1.upper()
    assert "AUTOMOTIVE CAST IRON DISC BRAKE ROTORS" in cleaned1.upper()

    raw2 = "SUPER KERNEL BASMATI RICE FULLY MILLED PARBOILED 50KG JUTE BAGS"
    cleaned2 = preprocess_cargo_text(raw2)
    assert "JUTE" not in cleaned2.upper()
    assert "BAGS" not in cleaned2.upper()
    assert "BASMATI RICE" in cleaned2.upper()

    raw3 = "TEMP SET -1.5C 1X40RF FRESH CITRUS MANDARINS PACKED IN 10KG CORRUGATED BOXES"
    cleaned3 = preprocess_cargo_text(raw3)
    assert "TEMP" not in cleaned3.upper()
    assert "CORRUGATED" not in cleaned3.upper()
    assert "BOXES" not in cleaned3.upper()
    assert "CITRUS MANDARINS" in cleaned3.upper()


def test_packaging_stopwords():
    text = "rice in corrugated boxes and jute bags palletized"
    keywords = extract_product_keywords(text)
    assert "rice" in keywords
    assert "boxes" not in keywords
    assert "bags" not in keywords
    assert "corrugated" not in keywords
    assert "palletized" not in keywords


def test_empty_input():
    assert preprocess_cargo_text("") == ""
    assert preprocess_cargo_text("   ") == ""


if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in test_funcs:
        try:
            test()
            print(f"  ✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

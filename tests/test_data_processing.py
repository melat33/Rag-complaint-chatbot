import pandas as pd
from src.data_processing import clean_text, filter_and_clean, NARRATIVE_COL, PRODUCT_COL


def test_clean_text_lowercases_and_strips_boilerplate():
    raw = "I am writing to file a complaint about my card ending XXXXXXXX1234."
    cleaned = clean_text(raw)
    assert cleaned == cleaned.lower()
    assert "writing to file a complaint" not in cleaned
    assert "xxxxxxxx1234" not in cleaned


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_filter_and_clean_drops_out_of_scope_products():
    df = pd.DataFrame({
        "Complaint ID": [1, 2, 3],
        PRODUCT_COL: ["Credit card", "Mortgage", "Personal loan"],
        NARRATIVE_COL: ["This card charged me twice for no reason.", "House issue.", "Loan rate changed suddenly."],
        "Issue": ["Billing", "Servicing", "Rate"],
        "Sub-issue": ["", "", ""],
        "Company": ["A", "B", "C"],
        "State": ["CA", "NY", "TX"],
        "Date received": ["2024-01-01"] * 3,
    })
    result = filter_and_clean(df)
    assert set(result[PRODUCT_COL]) == {"Credit card", "Personal loan"}


def test_filter_and_clean_drops_empty_narratives():
    df = pd.DataFrame({
        "Complaint ID": [1, 2],
        PRODUCT_COL: ["Credit card", "Credit card"],
        NARRATIVE_COL: ["This is a real narrative about a billing dispute.", ""],
        "Issue": ["Billing", "Billing"],
        "Sub-issue": ["", ""],
        "Company": ["A", "A"],
        "State": ["CA", "CA"],
        "Date received": ["2024-01-01", "2024-01-01"],
    })
    result = filter_and_clean(df)
    assert len(result) == 1

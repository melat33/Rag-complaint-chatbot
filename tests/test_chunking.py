import pandas as pd
from src.chunking import chunk_dataframe, stratified_sample


def test_chunk_dataframe_attaches_metadata():
    df = pd.DataFrame({
        "Complaint ID": [101],
        "Product": ["Credit card"],
        "Issue": ["Billing dispute"],
        "Sub-issue": ["Incorrect charge"],
        "Company": ["Acme Bank"],
        "State": ["CA"],
        "Date received": ["2024-01-01"],
        "cleaned_narrative": ["a" * 1200],  # long enough to force multiple chunks
    })
    chunks = chunk_dataframe(df)

    assert len(chunks) > 1
    assert all(c["complaint_id"] == 101 for c in chunks)
    assert all(c["product_category"] == "Credit card" for c in chunks)
    assert chunks[0]["total_chunks"] == len(chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_dataframe_short_text_produces_one_chunk():
    df = pd.DataFrame({
        "Complaint ID": [1],
        "Product": ["Personal loan"],
        "Issue": ["Rate"],
        "Sub-issue": [""],
        "Company": ["X"],
        "State": ["NY"],
        "Date received": ["2024-01-01"],
        "cleaned_narrative": ["short complaint text"],
    })
    chunks = chunk_dataframe(df)
    assert len(chunks) == 1


def test_stratified_sample_respects_max_size():
    df = pd.DataFrame({
        "Product": ["Credit card"] * 70 + ["Personal loan"] * 30,
        "value": range(100),
    })
    sampled = stratified_sample(df, sample_size=20)
    assert len(sampled) <= 22  # allow small rounding slack
    assert set(sampled["Product"]) == {"Credit card", "Personal loan"}

"""
Task 1: Exploratory Data Analysis & Preprocessing.

Reads the raw CFPB complaint export in chunks (memory-safe for large files or
low-RAM machines), maps product names to CrediTrust's four target categories
using keyword matching (CFPB has renamed several categories over time, so an
exact-string match silently drops rows), filters to complaints with a usable
narrative, cleans the text, and writes data/processed/filtered_complaints.csv.

Usage:
    python -m src.data_processing
"""
import re
import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

NEEDED_COLS = [
    "Complaint ID", "Product", "Issue", "Sub-issue",
    "Consumer complaint narrative", "Company", "State", "Date received",
]

CHUNK_SIZE = 50_000  # rows per chunk while streaming through the raw CSV

# Common CFPB boilerplate consumers paste in automatically-generated complaints.
BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint",
    r"this complaint is in regards to",
    r"to whom it may concern",
    r"xx/xx/\d{2,4}",     # CFPB's redacted date placeholders, e.g. XX/XX/2023
    r"x{2,}",             # redacted account/card numbers, e.g. XXXXXXXX1234
]


def _map_to_target_product(raw_product: str):
    """
    Map a raw CFPB product string to one of CrediTrust's four target categories
    via keyword matching, e.g. "Payday loan, title loan, personal loan, or
    advance loan" -> "Personal loan". Returns None if out of scope.
    """
    if not isinstance(raw_product, str):
        return None
    lowered = raw_product.lower()
    for canonical_name, keywords in config.TARGET_PRODUCT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return canonical_name
    return None


def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate/redaction placeholders, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = re.sub(r"[^a-z0-9.,!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply target-product mapping, narrative filtering, and text cleaning to one chunk."""
    chunk = chunk.copy()
    chunk["target_product"] = chunk[PRODUCT_COL].apply(_map_to_target_product)
    chunk = chunk[chunk["target_product"].notna()]

    chunk = chunk[chunk[NARRATIVE_COL].notna() & (chunk[NARRATIVE_COL].str.strip() != "")]

    chunk["cleaned_narrative"] = chunk[NARRATIVE_COL].apply(clean_text)
    chunk = chunk[chunk["cleaned_narrative"].str.len() > 20]

    chunk[PRODUCT_COL] = chunk["target_product"]
    return chunk


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-streaming variant of the same filtering + cleaning logic, for use on a
    dataframe that's already fully loaded in memory (e.g. in tests, or a small
    sample in a notebook). For the full raw file, use load_and_filter() instead
    -- it streams in chunks and never holds the whole file in memory at once.
    """
    return _clean_chunk(df).reset_index(drop=True)


def run_eda(df: pd.DataFrame) -> dict:
    """
    Non-streaming variant of run_eda_streaming(), for a dataframe already fully
    loaded in memory. For the full raw file, use run_eda_streaming() instead.
    """
    product_counts = df[PRODUCT_COL].value_counts()

    has_narrative = df[NARRATIVE_COL].notna() & (df[NARRATIVE_COL].str.strip() != "")
    with_narrative = int(has_narrative.sum())
    without_narrative = int((~has_narrative).sum())

    word_count = df.loc[has_narrative, NARRATIVE_COL].str.split().str.len().describe()

    return {
        "product_counts": product_counts,
        "with_narrative": with_narrative,
        "without_narrative": without_narrative,
        "word_count": word_count,
    }


def load_and_filter(path=None, chunksize: int = CHUNK_SIZE) -> pd.DataFrame:
    """
    Stream the full raw CSV in chunks, filtering + cleaning each chunk as it's
    read, and concatenate only the matches. Peak memory stays bounded by
    `chunksize` plus the (small) running total of filtered rows -- this is what
    makes it safe to run against the full multi-GB CFPB export on a low-RAM
    machine, and it also avoids the "first N rows only" bias you get from
    nrows=N when the file is sorted by date.
    """
    path = path or (config.DATA_RAW_DIR / "complaints.csv")
    logger.info(f"Streaming {path} in chunks of {chunksize:,} rows")

    matched_chunks = []
    total_rows_seen = 0

    reader = pd.read_csv(
        path, usecols=NEEDED_COLS, chunksize=chunksize,
        dtype={"Company": str}, low_memory=False, on_bad_lines="skip",
    )
    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)
        cleaned = _clean_chunk(chunk)
        if len(cleaned) > 0:
            matched_chunks.append(cleaned)
        if (i + 1) % 10 == 0:
            kept_so_far = sum(len(c) for c in matched_chunks)
            logger.info(f"  ...scanned {total_rows_seen:,} rows, kept {kept_so_far:,} so far")

    result = pd.concat(matched_chunks, ignore_index=True) if matched_chunks else pd.DataFrame()
    logger.info(f"Done. Scanned {total_rows_seen:,} rows total, kept {len(result):,} after filtering + cleaning.")

    keep_cols = [
        "Complaint ID", PRODUCT_COL, "Issue", "Sub-issue", "Company", "State",
        "Date received", NARRATIVE_COL, "cleaned_narrative",
    ]
    keep_cols = [c for c in keep_cols if c in result.columns]
    return result[keep_cols] if len(result) else result


def run_eda_streaming(path=None, chunksize: int = CHUNK_SIZE) -> dict:
    """
    Compute EDA stats (product distribution, narrative coverage, narrative
    word-count distribution) across the FULL raw file without ever loading it
    all into memory at once -- accumulates running totals chunk by chunk.
    """
    path = path or (config.DATA_RAW_DIR / "complaints.csv")
    logger.info(f"Streaming {path} for EDA")

    product_counts = pd.Series(dtype=int)
    with_narrative = 0
    without_narrative = 0
    word_counts = []

    reader = pd.read_csv(
        path, usecols=NEEDED_COLS, chunksize=chunksize,
        dtype={"Company": str}, low_memory=False, on_bad_lines="skip",
    )
    for chunk in reader:
        product_counts = product_counts.add(chunk[PRODUCT_COL].value_counts(), fill_value=0)

        has_narrative = chunk[NARRATIVE_COL].notna() & (chunk[NARRATIVE_COL].str.strip() != "")
        with_narrative += int(has_narrative.sum())
        without_narrative += int((~has_narrative).sum())

        word_counts.extend(
            chunk.loc[has_narrative, NARRATIVE_COL].str.split().str.len().tolist()
        )

    word_counts = pd.Series(word_counts)
    stats = {
        "product_counts": product_counts.astype(int).sort_values(ascending=False),
        "with_narrative": with_narrative,
        "without_narrative": without_narrative,
        "word_count": word_counts.describe(),
    }
    logger.info(f"With narrative: {with_narrative:,} | Without narrative: {without_narrative:,}")
    return stats


def main():
    stats = run_eda_streaming()

    print("\nProduct Counts (full raw dataset)")
    print(stats["product_counts"])

    print("\nNarrative Statistics")
    print(f"With narrative:    {stats['with_narrative']:,}")
    print(f"Without narrative: {stats['without_narrative']:,}")
    print(stats["word_count"])

    filtered = load_and_filter()

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(config.FILTERED_COMPLAINTS_PATH, index=False)
    logger.info(f"Saved {len(filtered):,} cleaned complaints to {config.FILTERED_COMPLAINTS_PATH}")


if __name__ == "__main__":
    main()
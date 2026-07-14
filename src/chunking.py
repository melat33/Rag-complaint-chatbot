"""
Task 2: Text chunking.

Long complaint narratives embed poorly as a single vector - the embedding
gets "diluted" across too many topics. We split each narrative into
overlapping chunks so each vector represents one coherent slice of text,
and keep enough overlap (50 chars) that we don't sever a sentence's meaning
right at a chunk boundary.
"""
import logging
from typing import List, Dict

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src import config

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    # Try to split on paragraph/sentence boundaries first, only falling back
    # to a hard character cut if nothing cleaner is available.
    separators=["\n\n", "\n", ". ", " ", ""],
)


def stratified_sample(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """
    Sample complaints proportionally by product, so no category is
    over/under-represented in the 10-15K training sample.
    """
    sample_size = sample_size or config.SAMPLE_SIZE
    n_products = df["Product"].nunique()

    sampled = (
        df.groupby("Product", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), max(1, int(sample_size * len(g) / len(df)))),
            random_state=config.RANDOM_SEED,
        ))
    )
    logger.info(f"Stratified sample: {len(sampled):,} rows across {n_products} products")
    return sampled.reset_index(drop=True)


def chunk_dataframe(df: pd.DataFrame, text_col: str = "cleaned_narrative") -> List[Dict]:
    """
    Chunk every narrative in the dataframe and attach the metadata fields
    the assignment requires (complaint_id, product, issue, chunk_index, etc.)
    so each chunk can be traced back to its source complaint.
    """
    records = []
    id_col = "Complaint ID" if "Complaint ID" in df.columns else df.columns[0]

    for _, row in df.iterrows():
        chunks = _splitter.split_text(row[text_col])
        for i, chunk in enumerate(chunks):
            records.append({
                "chunk_text": chunk,
                "complaint_id": row[id_col],
                "product_category": row.get("Product"),
                "issue": row.get("Issue"),
                "sub_issue": row.get("Sub-issue"),
                "company": row.get("Company"),
                "state": row.get("State"),
                "date_received": row.get("Date received"),
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
    logger.info(f"Produced {len(records):,} chunks from {len(df):,} complaints "
                f"(avg {len(records) / max(len(df), 1):.1f} chunks/complaint)")
    return records

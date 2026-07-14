"""
Task 2: Stratified sampling, chunking, embedding, and vector store indexing.

Takes data/processed/filtered_complaints.csv (from Task 1), draws a proportional
stratified sample, splits narratives into overlapping chunks, embeds each chunk,
and indexes everything into ChromaDB at vector_store/chroma_db/.

Usage:
    python -m src.build_index
"""
import logging

import pandas as pd

from src import config
from src.chunking import stratified_sample, chunk_dataframe
from src.embedding import embed_and_index, get_chroma_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Loading {config.FILTERED_COMPLAINTS_PATH}")
    df = pd.read_csv(config.FILTERED_COMPLAINTS_PATH)
    logger.info(f"Loaded {len(df):,} cleaned complaints")

    sample_df = stratified_sample(df)
    logger.info(f"Stratified sample: {len(sample_df):,} complaints")
    print(sample_df["Product"].value_counts())

    chunk_records = chunk_dataframe(sample_df)
    logger.info(f"Produced {len(chunk_records):,} chunks "
                f"(avg {len(chunk_records) / len(sample_df):.2f} chunks/complaint)")

    embed_and_index(chunk_records)

    collection = get_chroma_collection()
    logger.info(f"Vector store ready: {collection.count():,} chunks indexed at "
                f"{config.CHROMA_PERSIST_DIR}")

    # Sanity check: run one test query so you can confirm retrieval works before
    # moving to Task 3.
    from src.embedding import get_embedding_model
    model = get_embedding_model()
    test_query = "unauthorized charges on my credit card"
    results = collection.query(query_embeddings=model.encode([test_query]).tolist(), n_results=3)
    print(f"\nTest query: {test_query!r}")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{meta['product_category']}] similarity={1-dist:.3f}  id={meta['complaint_id']}")
        print(f"  {doc[:150]}...")


if __name__ == "__main__":
    main()
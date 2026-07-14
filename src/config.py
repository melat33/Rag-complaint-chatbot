"""
Single source of truth for paths, model names, and pipeline hyperparameters.
Every other module imports from here instead of hardcoding values, so a
change (e.g. swapping embedding models) only happens in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
FILTERED_COMPLAINTS_PATH = DATA_PROCESSED_DIR / "filtered_complaints.csv"

CHROMA_PERSIST_DIR = str(BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "vector_store/chroma_db"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "complaint_chunks")

# --- Product scope (Task 1) ---
# CFPB's product taxonomy has changed over time (e.g. "Personal loan" is now folded
# into "Payday loan, title loan, personal loan, or advance loan"). Matching on a
# substring keyword, rather than an exact product-name string, keeps the filter
# working even as CFPB renames categories.
TARGET_PRODUCT_KEYWORDS = {
    "Credit card": ["credit card"],
    "Personal loan": ["personal loan", "payday loan", "title loan"],
    "Savings account": ["savings account", "checking or savings"],
    "Money transfer": ["money transfer"],
}
TARGET_PRODUCTS = list(TARGET_PRODUCT_KEYWORDS.keys())  # kept for backward compatibility

# --- Chunking (Task 2) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding (Task 2) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384

# --- Retrieval (Task 3) ---
TOP_K = int(os.getenv("TOP_K", 5))

# --- Generation (Task 3) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.2       # low temperature: we want grounded, consistent analyst answers
LLM_MAX_TOKENS = 600

# --- Sampling (Task 2 stratified sample) ---
SAMPLE_SIZE = 2000
RANDOM_SEED = 42

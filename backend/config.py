"""
Central configuration for the Knowledge Desk assistant.

All tunable values live here so the rest of the codebase never hardcodes
model names, chunk sizes, or file paths.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Hugging Face settings ---------------------------------------------------
# A default token is optional: the frontend also lets a user paste their own
# token in the sidebar, which is sent along with each /api/ask request and
# takes priority over this one. Setting HF_API_KEY here just gives the
# server a fallback so it works without per-request tokens too.
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

# --- Chunking -----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120")) # overlap between chunks

# --- Retrieval ------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))                   # chunks retrieved per question

# --- Storage paths ----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "documents.json"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Words stripped out before TF-IDF scoring — common enough to carry no
# retrieval signal on their own.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "this", "that", "these", "those", "at", "by", "from", "into",
    "about", "than", "so", "not", "no", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "may", "might", "must",
    "you", "your", "i", "we", "our", "they", "their", "he", "she", "him",
    "her", "his", "them", "what", "which", "who", "whom", "when", "where",
    "why", "how", "there", "here",
}

import os
import re

# ---- Google Gemini Model ----
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")

# ---- Paths ----
SOURCE_DOC_DIR = os.getenv("SOURCE_DOC_DIR", "./pdfs")
IMAGE_DIR = os.getenv("IMAGE_DIR", "./images")
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

# ---- App Settings ----
AGENT_NAME = re.sub(
    r"\s+", "_",
    re.sub(r"[^a-zA-Z]", " ", os.getenv("AGENT_NAME", "rag_agent"))
).lower()

TOP_K = int(os.getenv("TOP_K", "3"))

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:8000"
).split(",")

ORG_NAME = os.getenv("Hasnat Samiul", "Samiul")
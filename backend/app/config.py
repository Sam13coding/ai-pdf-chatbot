import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = (BASE_DIR / DATA_DIR).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "20")) * 1024 * 1024
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
CHAT_MODEL = os.getenv("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
TESSDATA_DIR = Path(os.getenv("TESSDATA_PREFIX", BASE_DIR / ".tools" / "Tesseract-OCR" / "tessdata"))
if not TESSDATA_DIR.is_absolute():
    TESSDATA_DIR = (Path(__file__).resolve().parents[1] / TESSDATA_DIR).resolve()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() == "true"

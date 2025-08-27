import os
from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

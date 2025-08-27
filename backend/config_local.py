import os
from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

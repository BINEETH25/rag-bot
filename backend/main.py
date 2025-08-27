from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .rag_local import answer
from .config_local import API_HOST, API_PORT

app = FastAPI(title="RAG API (Local/Ollama)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    k: int = 4

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(req: ChatRequest):
    return answer(req.query, req.k)

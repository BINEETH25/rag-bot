# --- replace the imports at the top ---
from typing import List, Optional
import os
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import Document
from .config_local import CHROMA_DIR, DATA_DIR, OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL


# --- add this helper to load docs without 'unstructured' ---
def load_docs() -> List[Document]:
    docs: List[Document] = []
    data_dir = Path(DATA_DIR)
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                docs.extend(loader.load())
            elif path.suffix.lower() in {".txt", ".log"}:
                loader = TextLoader(str(path), encoding="utf-8")
                docs.extend(loader.load())
            elif path.suffix.lower() in {".md", ".markdown"}:
                loader = UnstructuredMarkdownLoader(str(path))
                docs.extend(loader.load())
            else:
                # skip other types for now to avoid extra deps
                print(f"[skip] {path.name} (unsupported type)")
        except Exception as e:
            print(f"[error] {path.name}: {e}")
    return docs


def chunk_docs(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_vectorstore():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())


def ingest():
    raw_docs = load_docs()
    if not raw_docs:
        raise RuntimeError(
            "No documents loaded. Put PDFs/TXTs/MDs into the 'data/' folder, "
            "or check the console for [error]/[skip] lines."
        )
    chunks = chunk_docs(raw_docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    return len(raw_docs), len(chunks)


def answer(query: str, k: int = 4, system_prompt: Optional[str] = None) -> dict:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    ctx_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([
        f"[{i+1}] {d.metadata.get('source','?')}\n{d.page_content}"
        for i, d in enumerate(ctx_docs)
    ])

    sys = system_prompt or (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the answer isn't in the context, say you don't know."
    )

    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    resp = llm.invoke(messages)

    sources = [{"source": d.metadata.get("source", ""), "score": getattr(d, 'score', None)}
               for d in ctx_docs]
    return {"answer": resp.content, "sources": sources}

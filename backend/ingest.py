# backend/ingest.py
from backend.rag_local import ingest


if __name__ == "__main__":
    n_files, n_chunks = ingest()
    print(f"Ingested {n_files} files into {n_chunks} chunks.")

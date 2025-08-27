from typing import List, Optional
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from .config import CHROMA_DIR, DATA_DIR, OPENAI_API_KEY, EMBED_MODEL, CHAT_MODEL

def get_loader():
    # Load many file types from /data
    return DirectoryLoader(
        DATA_DIR,
        glob="**/*",
        loader_cls=UnstructuredFileLoader,
        show_progress=True
    )

def chunk_docs(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def get_embeddings():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY missing. Set in .env.")
    return OpenAIEmbeddings(model=EMBED_MODEL)

def get_vectorstore():
    embeddings = get_embeddings()
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def ingest():
    loader = get_loader()
    raw_docs = loader.load()
    chunks = chunk_docs(raw_docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    return len(raw_docs), len(chunks)

def answer(query: str, k: int = 4, system_prompt: Optional[str] = None) -> dict:
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    ctx_docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([f"[{i+1}] {d.metadata.get('source','?')}\n{d.page_content}"
                           for i, d in enumerate(ctx_docs)])

    sys = system_prompt or (
        "You are a helpful assistant. Answer using only the provided context. "
        "If the answer isn't in the context, say you don't know."
    )

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    resp = llm.invoke(messages)

    sources = [{"source": d.metadata.get("source", ""), "score": getattr(d, 'score', None)} for d in ctx_docs]
    return {"answer": resp.content, "sources": sources}

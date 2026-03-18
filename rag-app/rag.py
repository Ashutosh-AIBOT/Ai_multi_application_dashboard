import hashlib, time
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import httpx
from config import (GROQ_API_KEY, GEMINI_API_KEY, LOCAL_LLM_URL,
                    LOCAL_MODEL, CHROMA_PATH, EMBED_MODEL)
from typing import List

# ── lazy singletons ────────────────────────────────────
_embedder = None
_reranker = None
_chroma   = None


def embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-2-v2", max_length=256)
    return _reranker


def chroma():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma


def collection(user_id: str, doc_id: str):
    name = "u" + hashlib.md5(user_id.encode()).hexdigest()[:8] + \
           "d" + hashlib.md5(doc_id.encode()).hexdigest()[:8]
    return chroma().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "user": user_id, "doc": doc_id}
    )


# ── chunking ───────────────────────────────────────────
def chunk_text(text: str, size: int = 80, overlap: int = 15) -> List[str]:
    words, chunks, i = text.split(), [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


# ── ingest ─────────────────────────────────────────────
def ingest(text: str, doc_id: str, user_id: str,
           filename: str = "", chunk_size: int = 80) -> dict:
    chunks = chunk_text(text, chunk_size)
    emb    = embedder()
    vecs   = emb.encode(chunks, batch_size=32, show_progress_bar=False)
    col    = collection(user_id, doc_id)
    col.upsert(
        ids        = [f"{doc_id}_{i}" for i in range(len(chunks))],
        documents  = chunks,
        embeddings = vecs.tolist(),
        metadatas  = [{"doc_id": doc_id, "chunk_no": i,
                        "user_id": user_id, "file": filename}
                      for i in range(len(chunks))]
    )
    return {"doc_id": doc_id, "chunks": len(chunks)}


# ── retrieve + rerank ──────────────────────────────────
def retrieve(query: str, user_id: str, doc_ids: List[str],
             top_k: int = 5) -> List[dict]:
    emb  = embedder()
    qvec = emb.encode(query).tolist()
    cands = []

    for doc_id in doc_ids:
        try:
            col = collection(user_id, doc_id)
            if col.count() == 0:
                continue
            res = col.query(
                query_embeddings=[qvec],
                n_results=min(20, col.count())
            )
            for i, doc in enumerate(res["documents"][0]):
                cands.append({
                    "text"    : doc,
                    "doc_id"  : doc_id,
                    "chunk_no": res["metadatas"][0][i].get("chunk_no", i),
                    "score"   : round(1 - res["distances"][0][i], 4),
                })
        except Exception:
            continue

    if not cands:
        return []

    # rerank
    rr     = reranker()
    scores = rr.predict([[query, c["text"]] for c in cands])
    for i, c in enumerate(cands):
        c["rerank_score"] = round(float(scores[i]), 4)
    cands.sort(key=lambda x: x["rerank_score"], reverse=True)
    return cands[:top_k]


# ── LangChain RAG chain ────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't have that info."

Context:
{context}

Question: {question}

Answer:""")


def _get_llm():
    if GROQ_API_KEY:
        try:
            return ChatGroq(api_key=GROQ_API_KEY,
                            model_name="llama-3.3-70b-versatile",
                            temperature=0.3)
        except Exception:
            pass
    if GEMINI_API_KEY:
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.3)
        except Exception:
            pass
    return None  # will use local


async def rag_answer(query: str, user_id: str,
                     doc_ids: List[str], top_k: int = 5) -> dict:
    t0     = time.time()
    chunks = retrieve(query, user_id, doc_ids, top_k)

    if not chunks:
        return {"answer": "No relevant documents found.",
                "chunks": [], "provider": "none",
                "total_ms": round((time.time() - t0) * 1000)}

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    llm     = _get_llm()

    if llm:
        chain  = RAG_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})
        prov   = "groq" if isinstance(llm, ChatGroq) else "gemini"
    else:
        # local fallback
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{LOCAL_LLM_URL}/api/generate",
                json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False})
            answer = r.json()["response"]
        prov = "local"

    return {
        "answer"  : answer,
        "provider": prov,
        "chunks"  : chunks,
        "total_ms": round((time.time() - t0) * 1000),
    }


def list_docs(user_id: str) -> List[dict]:
    cols = chroma().list_collections()
    docs = []
    for c in cols:
        if c.metadata.get("user") == user_id:
            col = chroma().get_collection(c.name)
            docs.append({
                "doc_id": c.metadata.get("doc"),
                "chunks": col.count(),
            })
    return docs

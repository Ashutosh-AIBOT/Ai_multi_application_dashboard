from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from config import API_KEY
from rag import ingest, rag_answer, list_docs

app = FastAPI(title="RAG App", description="Upload docs → query with reranking")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(security)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key


class QueryReq(BaseModel):
    query  : str
    user_id: str
    doc_ids: List[str] = []
    top_k  : int = 5


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag-app"}


@app.post("/upload")
async def upload(
    file      : UploadFile = File(...),
    user_id   : str = "default",
    chunk_size: int = 80,
    _         : str = Depends(verify),
):
    content = await file.read()
    fname   = file.filename

    if fname.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
    elif fname.endswith(".pdf"):
        try:
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception as e:
            raise HTTPException(400, f"PDF error: {e}")
    else:
        raise HTTPException(400, "Only .txt and .pdf supported")

    doc_id = fname.replace(" ", "_").replace(".", "_")
    result = ingest(text, doc_id, user_id, fname, chunk_size)
    return {"ok": True, **result, "user_id": user_id}


@app.post("/query")
async def query(req: QueryReq, _=Depends(verify)):
    if not req.doc_ids:
        docs       = list_docs(req.user_id)
        req.doc_ids = [d["doc_id"] for d in docs]
    result = await rag_answer(req.query, req.user_id, req.doc_ids, req.top_k)
    return result


@app.get("/docs/{user_id}")
async def docs(user_id: str, _=Depends(verify)):
    return {"docs": list_docs(user_id), "user_id": user_id}

import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from config import API_KEY
from papers import (search_arxiv, search_semantic,
                    summarize_paper, compare_papers)

app = FastAPI(title="Research App",
              description="Search + summarise research papers")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(security)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key


class SearchReq(BaseModel):
    query      : str
    max_results: int = 5
    source     : str = "both"   # arxiv | semantic_scholar | both

class CompareReq(BaseModel):
    paper_ids: List[str]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "research-app",
            "sources": ["arxiv", "semantic_scholar"]}


@app.post("/search")
async def search(req: SearchReq, _=Depends(verify)):
    if req.source == "arxiv":
        papers = await search_arxiv(req.query, req.max_results)
    elif req.source == "semantic_scholar":
        papers = await search_semantic(req.query, req.max_results)
    else:
        # both in parallel
        arxiv_r, sem_r = await asyncio.gather(
            search_arxiv(req.query,   req.max_results // 2 + 1),
            search_semantic(req.query, req.max_results // 2 + 1),
        )
        papers = (arxiv_r + sem_r)[:req.max_results]

    return {"papers": papers, "count": len(papers), "query": req.query}


@app.get("/summarize/{paper_id}")
async def summarize(paper_id: str, _=Depends(verify)):
    result = await summarize_paper(paper_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.post("/compare")
async def compare(req: CompareReq, _=Depends(verify)):
    if len(req.paper_ids) < 2:
        raise HTTPException(400, "Need at least 2 paper IDs")
    result = await compare_papers(req.paper_ids)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/trending/{topic}")
async def trending(topic: str, _=Depends(verify)):
    papers = await search_arxiv(f"{topic} survey 2024", 8)
    return {"topic": topic, "papers": papers, "count": len(papers)}

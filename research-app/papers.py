import asyncio, time, urllib.parse, httpx
from config import GROQ_API_KEY, GEMINI_API_KEY, LOCAL_LLM_URL, LOCAL_MODEL
from typing import List


async def _ask(prompt: str, system: str = "You are a research analyst."):
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile",
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user",   "content": prompt}],
                          "max_tokens": 1024})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"], "groq"
        except Exception:
            pass
    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                    json={"contents": [{"parts": [{"text": f"{system}\n\n{prompt}"}]}]})
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"], "gemini"
        except Exception:
            pass
    return "LLM not available", "none"


async def search_arxiv(query: str, max_results: int = 5) -> List[dict]:
    q   = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{q}&max_results={max_results}&sortBy=relevance")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        try:
            r = await c.get(url)
            r.raise_for_status()
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            ns   = {"a": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("a:entry", ns):
                pid = entry.find("a:id", ns).text.split("/abs/")[-1]
                papers.append({
                    "id"       : pid,
                    "title"    : entry.find("a:title", ns).text.strip(),
                    "abstract" : entry.find("a:summary", ns).text.strip()[:600],
                    "authors"  : [a.find("a:name", ns).text
                                  for a in entry.findall("a:author", ns)][:4],
                    "published": entry.find("a:published", ns).text[:10],
                    "url"      : f"https://arxiv.org/abs/{pid}",
                    "pdf_url"  : f"https://arxiv.org/pdf/{pid}",
                    "source"   : "arxiv",
                })
            return papers
        except Exception as e:
            print(f"ArXiv error: {e}")
            return []


async def search_semantic(query: str, max_results: int = 5) -> List[dict]:
    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": max_results,
              "fields": "title,abstract,authors,year,citationCount,url"}
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
        try:
            r = await c.get(url, params=params)
            r.raise_for_status()
            papers = []
            for p in r.json().get("data", []):
                papers.append({
                    "id"       : p.get("paperId", ""),
                    "title"    : p.get("title", ""),
                    "abstract" : (p.get("abstract") or "")[:600],
                    "authors"  : [a["name"] for a in p.get("authors", [])][:4],
                    "year"     : p.get("year", ""),
                    "citations": p.get("citationCount", 0),
                    "url"      : p.get("url", ""),
                    "source"   : "semantic_scholar",
                })
            return papers
        except Exception as e:
            print(f"Semantic Scholar error: {e}")
            return []


SUMMARIZE_PROMPT = """Summarise this research paper:
Title: {title}
Authors: {authors}
Abstract: {abstract}

Return:
ONE LINE SUMMARY:
KEY CONTRIBUTIONS:
- 
METHODOLOGY:
LIMITATIONS:
-
WHO SHOULD READ THIS:"""

COMPARE_PROMPT = """Compare these papers:
{papers_text}

Return:
COMMON THEMES:
KEY DIFFERENCES:
MOST PRACTICAL:
READING ORDER:"""


async def summarize_paper(paper_id: str) -> dict:
    t0     = time.time()
    papers = await search_arxiv(f"id:{paper_id}", 1)
    if not papers:
        return {"error": "Paper not found"}
    p       = papers[0]
    summary, prov = await _ask(
        SUMMARIZE_PROMPT.format(
            title    = p["title"],
            authors  = ", ".join(p["authors"]),
            abstract = p["abstract"]),
        "You are a precise academic paper summariser.")
    return {"paper": p, "summary": summary, "provider": prov,
            "total_ms": round((time.time() - t0) * 1000)}


async def compare_papers(paper_ids: List[str]) -> dict:
    t0     = time.time()
    papers = []
    for pid in paper_ids[:4]:
        found = await search_arxiv(f"id:{pid}", 1)
        if found:
            papers.append(found[0])
    if len(papers) < 2:
        return {"error": "Need at least 2 valid papers"}
    papers_text = "\n\n".join(
        f"Paper {i+1}: {p['title']}\n{p['abstract']}"
        for i, p in enumerate(papers))
    comparison, prov = await _ask(
        COMPARE_PROMPT.format(papers_text=papers_text))
    return {"papers": papers, "comparison": comparison,
            "provider": prov, "total_ms": round((time.time() - t0) * 1000)}
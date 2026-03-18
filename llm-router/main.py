from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from config import API_KEY, GROQ_MODELS, GEMINI_MODELS, LOCAL_MODEL
from llm import smart_chat, state

app = FastAPI(title="LLM Router", description="Groq → Gemini → Local fallback")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(security)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key


class ChatReq(BaseModel):
    message    : str
    system     : str   = "You are a helpful assistant."
    temperature: float = 0.7
    model      : Optional[str] = None


@app.get("/health")
async def health():
    return {
        "status"  : "ok",
        "provider": state["provider"],
        "groq_key": state["groq_ok"],
        "gem_key" : state["gemini_ok"],
    }


@app.post("/chat")
async def chat(req: ChatReq, _=Depends(verify)):
    try:
        result = await smart_chat(
            message    = req.message,
            system     = req.system,
            temperature= req.temperature,
            model      = req.model,
        )
        return result
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/models")
async def models(_=Depends(verify)):
    return {"groq": GROQ_MODELS, "gemini": GEMINI_MODELS, "local": [LOCAL_MODEL]}


@app.get("/status")
async def status(_=Depends(verify)):
    return state

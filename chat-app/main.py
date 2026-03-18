from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from config import API_KEY
from memory import (chat, get_session, get_memory,
                    add_memory, SESSIONS, MEMORY)

app = FastAPI(title="Chat App", description="Personalised chat with memory")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(security)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key


class ChatReq(BaseModel):
    message : str
    user_id : str
    session : str    = "default"
    system  : Optional[str] = None
    remember: Optional[str] = None


class MemoryReq(BaseModel):
    fact: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "chat-app",
            "sessions": len(SESSIONS), "users": len(MEMORY)}


@app.post("/chat")
async def chat_endpoint(req: ChatReq, _=Depends(verify)):
    try:
        return await chat(req.message, req.user_id,
                          req.session, req.system, req.remember)
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/history/{uid}/{sid}")
async def history(uid: str, sid: str, _=Depends(verify)):
    return {"history": get_session(uid, sid), "user_id": uid, "session": sid}


@app.get("/memory/{uid}")
async def memory(uid: str, _=Depends(verify)):
    return {"memory": get_memory(uid), "user_id": uid,
            "count": len(get_memory(uid))}


@app.post("/memory/{uid}")
async def add(uid: str, req: MemoryReq, _=Depends(verify)):
    add_memory(uid, req.fact)
    return {"added": req.fact, "total": len(get_memory(uid))}


@app.delete("/memory/{uid}")
async def clear_memory(uid: str, _=Depends(verify)):
    MEMORY[uid] = []
    return {"cleared": True}


@app.delete("/history/{uid}/{sid}")
async def clear_session(uid: str, sid: str, _=Depends(verify)):
    SESSIONS[f"{uid}:{sid}"] = []
    return {"cleared": True}

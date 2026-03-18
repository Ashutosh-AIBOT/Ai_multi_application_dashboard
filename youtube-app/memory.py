import time, httpx
from config import GROQ_API_KEY, GEMINI_API_KEY, LOCAL_LLM_URL, LOCAL_MODEL, MAX_HISTORY
from typing import Dict, List, Optional

SESSIONS: Dict[str, List[dict]] = {}
MEMORY:   Dict[str, List[str]]  = {}

FACT_TRIGGERS = ["my name is","i am called","i work","i live","i like",
                 "i love","i hate","i prefer","my job","i'm a","i study"]

async def _ask(messages: list, system: str) -> tuple:
    # Try Groq
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": "llama-3.3-70b-versatile",
                          "messages": [{"role":"system","content":system}] + messages,
                          "max_tokens": 1024}
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"], "groq"
        except Exception:
            pass
    # Try Gemini
    if GEMINI_API_KEY:
        try:
            full = system + "\n\n" + "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in messages)
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                    json={"contents":[{"parts":[{"text":full}]}]}
                )
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"], "gemini"
        except Exception:
            pass
    # Local fallback
    async with httpx.AsyncClient(timeout=120) as c:
        prompt = system + "\n\n" + "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in messages)
        r = await c.post(f"{LOCAL_LLM_URL}/api/generate",
            json={"model": LOCAL_MODEL, "prompt": prompt, "stream": False})
        return r.json()["response"], "local"

def get_session(uid, sid):
    k = f"{uid}:{sid}"
    if k not in SESSIONS: SESSIONS[k] = []
    return SESSIONS[k]

def get_memory(uid):
    if uid not in MEMORY: MEMORY[uid] = []
    return MEMORY[uid]

def add_memory(uid, fact):
    m = get_memory(uid)
    if fact not in m: m.append(fact)

def _build_system(uid, custom):
    base = custom or "You are a helpful, friendly AI assistant."
    mem  = get_memory(uid)
    if mem:
        facts = "\n".join(f"- {f}" for f in mem[-10:])
        return f"{base}\n\nThings you know about this user:\n{facts}"
    return base

async def chat(message, uid, sid, system=None, remember=None):
    t0      = time.time()
    session = get_session(uid, sid)
    sys_msg = _build_system(uid, system)
    if remember: add_memory(uid, remember)
    # auto extract facts
    for t in FACT_TRIGGERS:
        if t in message.lower():
            add_memory(uid, message.strip())
            break
    # build messages list
    msgs = [{"role": h["role"], "content": h["content"]}
            for h in session[-MAX_HISTORY:]]
    msgs.append({"role": "user", "content": message})
    reply, prov = await _ask(msgs, sys_msg)
    ts = time.time()
    session.append({"role": "user",      "content": message, "ts": ts})
    session.append({"role": "assistant", "content": reply,   "ts": ts})
    return {"reply": reply, "provider": prov, "session_id": sid,
            "history_len": len(session), "memory_count": len(get_memory(uid)),
            "total_ms": round((time.time()-t0)*1000)}
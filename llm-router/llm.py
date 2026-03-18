import httpx
from config import GROQ_API_KEY, GEMINI_API_KEY, LOCAL_LLM_URL, LOCAL_MODEL

state = {"provider": "groq", "groq_ok": bool(GROQ_API_KEY), "gemini_ok": bool(GEMINI_API_KEY)}

async def smart_chat(message, system="You are a helpful assistant.", temperature=0.7, model=None):
    errors = []

    # 1. Try Groq
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={"model": model or "llama-3.3-70b-versatile",
                          "messages": [{"role":"system","content":system},
                                       {"role":"user","content":message}],
                          "max_tokens": 1024, "temperature": temperature}
                )
                r.raise_for_status()
                state["provider"] = "groq"
                return {"reply": r.json()["choices"][0]["message"]["content"],
                        "provider": "groq", "model": model or "llama-3.3-70b-versatile"}
        except Exception as e:
            errors.append(f"groq: {e}")
            state["provider"] = "gemini"

    # 2. Try Gemini
    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                    json={"contents":[{"parts":[{"text":f"{system}\n\n{message}"}]}],
                          "generationConfig":{"maxOutputTokens":1024}}
                )
                r.raise_for_status()
                state["provider"] = "gemini"
                return {"reply": r.json()["candidates"][0]["content"]["parts"][0]["text"],
                        "provider": "gemini", "model": "gemini-2.5-flash"}
        except Exception as e:
            errors.append(f"gemini: {e}")
            state["provider"] = "local"

    # 3. Local fallback
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{LOCAL_LLM_URL}/api/generate",
                json={"model": LOCAL_MODEL, "prompt": f"{system}\n\n{message}", "stream": False})
            r.raise_for_status()
            state["provider"] = "local"
            return {"reply": r.json()["response"], "provider": "local", "model": LOCAL_MODEL}
    except Exception as e:
        errors.append(f"local: {e}")

    raise Exception(f"All providers failed: {errors}")
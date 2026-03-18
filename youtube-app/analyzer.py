import subprocess, json, re, time, tempfile, glob, httpx
from config import GROQ_API_KEY, GEMINI_API_KEY, LOCAL_LLM_URL, LOCAL_MODEL


async def _ask(prompt: str, system: str = "You are a helpful analyst."):
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
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{LOCAL_LLM_URL}/api/generate",
            json={"model": LOCAL_MODEL, "prompt": f"{system}\n\n{prompt}", "stream": False})
        return r.json()["response"], "local"


def get_video_info(url: str) -> dict:
    try:
        r = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-playlist", url],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            d = json.loads(r.stdout)
            return {
                "title"      : d.get("title", ""),
                "duration"   : d.get("duration", 0),
                "channel"    : d.get("channel", ""),
                "view_count" : d.get("view_count", 0),
                "upload_date": d.get("upload_date", ""),
                "description": (d.get("description") or "")[:400],
            }
    except Exception:
        pass
    return {}


def get_transcript(url: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["yt-dlp", "--write-auto-subs", "--sub-format", "vtt",
                 "--skip-download", "-o", f"{tmp}/v", url],
                capture_output=True, text=True, timeout=60)
            files = glob.glob(f"{tmp}/*.vtt")
            if files:
                raw   = open(files[0]).read()
                clean = re.sub(r"<[^>]+>", "", raw)
                clean = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}.*?-->\s*\d{2}:\d{2}:\d{2}\.\d{3}", "", clean)
                clean = re.sub(r"\n+", "\n", clean).strip()
                return clean[:4000]
    except Exception:
        pass
    return ""


def get_playlist_videos(url: str, max_videos: int = 10) -> list:
    try:
        r = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-json",
             "--playlist-end", str(max_videos), url],
            capture_output=True, text=True, timeout=30)
        videos = []
        for line in r.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                d = json.loads(line)
                videos.append({
                    "title"   : d.get("title", ""),
                    "id"      : d.get("id", ""),
                    "url"     : f"https://youtube.com/watch?v={d.get('id','')}",
                    "duration": d.get("duration", 0),
                })
            except Exception:
                pass
        return videos
    except Exception:
        return []


EMOTIONS = {
    "excited": ["wow","amazing","incredible","awesome","brilliant","fantastic"],
    "happy"  : ["great","wonderful","love","enjoy","happy","excellent","good"],
    "focused": ["important","note","key","must","learn","understand","critical"],
    "sad"    : ["unfortunately","problem","fail","bad","difficult","hard","sad"],
}

def detect_emotions(transcript: str) -> list:
    if not transcript:
        return []
    words, segments, window = transcript.lower().split(), [], 60
    for i in range(0, min(len(words), window * 8), window):
        chunk  = words[i:i+window]
        scores = {e: sum(1 for w in chunk if w in kws) for e, kws in EMOTIONS.items()}
        dominant = max(scores, key=scores.get) if max(scores.values(), default=0) > 0 else "neutral"
        segments.append({
            "segment"   : i // window + 1,
            "word_range": f"{i}-{i+window}",
            "emotion"   : dominant,
            "confidence": round(min(1.0, scores.get(dominant, 0) / 3), 2),
        })
    return segments


async def analyse_video(url: str) -> dict:
    t0   = time.time()
    info = get_video_info(url)
    tx   = get_transcript(url) or info.get("description", "")
    result = {
        "url"               : url,
        "video_info"        : info,
        "emotions"          : detect_emotions(tx),
        "transcript_preview": tx[:300] + "..." if len(tx) > 300 else tx,
    }
    if tx:
        content = tx[:2500]
        title   = info.get("title", "")
        summary, prov = await _ask(
            f"""Analyse this video and return:
SUMMARY: (3-4 sentences)
KEY POINTS:
- point 1
- point 2
- point 3
PROS:
+ pro 1
+ pro 2
CONS:
- con 1
- con 2
TARGET AUDIENCE: (one sentence)

Title: {title}
Content: {content}""",
            "You are a precise video content analyst.")
        result["analysis"] = summary
        result["provider"] = prov

        roadmap, _ = await _ask(
            f"Extract a step-by-step learning roadmap from this content. Number each step.\n\nTitle: {title}\nContent: {content}",
            "You extract structured learning paths.")
        result["roadmap"] = roadmap
    else:
        result["analysis"] = "No transcript available."
        result["roadmap"]  = "No transcript available."
        result["provider"] = "none"

    result["total_ms"] = round((time.time() - t0) * 1000)
    return result
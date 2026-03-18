import time
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import API_KEY
from analyzer import analyse_video, get_playlist_videos

app = FastAPI(title="YouTube App", description="Analyse videos, emotions, summaries")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = APIKeyHeader(name="X-API-Key")
def verify(key: str = Depends(security)):
    if key != API_KEY:
        raise HTTPException(401, "Wrong API Key")
    return key

JOBS: dict = {}


class VideoReq(BaseModel):
    url: str

class PlaylistReq(BaseModel):
    url       : str
    max_videos: int = 10


async def _run_job(job_id: str, url: str):
    JOBS[job_id]["status"] = "processing"
    try:
        result = await analyse_video(url)
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.get("/health")
async def health():
    try:
        import subprocess
        v = subprocess.run(["yt-dlp", "--version"],
                           capture_output=True, text=True)
        yt = v.stdout.strip()
    except Exception:
        yt = "not installed"
    return {"status": "ok", "service": "youtube-app", "yt_dlp": yt}


@app.post("/analyze")
async def analyze(req: VideoReq, bg: BackgroundTasks, _=Depends(verify)):
    """Start async video analysis job"""
    job_id = f"job_{int(time.time()*1000)}"
    JOBS[job_id] = {"status": "queued", "url": req.url}
    bg.add_task(_run_job, job_id, req.url)
    return {"job_id": job_id, "status": "queued"}


@app.get("/job/{job_id}")
async def get_job(job_id: str, _=Depends(verify)):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]


@app.post("/playlist")
async def playlist(req: PlaylistReq, _=Depends(verify)):
    """Get all video URLs from a YouTube playlist"""
    videos = get_playlist_videos(req.url, req.max_videos)
    return {"videos": videos, "count": len(videos)}


@app.get("/jobs")
async def list_jobs(_=Depends(verify)):
    return {k: v.get("status") for k, v in JOBS.items()}

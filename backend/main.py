"""Keylight local backend — FastAPI on localhost:8000.

Everything runs on this machine. The browser POSTs the chosen video to *this*
local process (that is not a cloud upload). The only optional network call is
Claude, for picking which subject matters; the pipeline works without it.

Run:  cd backend && python main.py          (serves the built UI too, if present)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    for p in (ROOT / ".env", ROOT.parent / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from jobs.manager import JobManager  # noqa: E402
from pipeline import perceive, understand  # noqa: E402

OUTPUTS = Path(os.environ.get("KEYLIGHT_OUTPUTS", ROOT / "outputs"))
SAMPLES = Path(os.environ.get("KEYLIGHT_SAMPLES", ROOT.parent / "samples"))
DIST = ROOT.parent / "frontend" / "dist"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
SERVABLE = {"original.mp4", "enhanced.mp4", "attention.mp4", "metadata.json"}

app = FastAPI(title="Keylight")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
jobs = JobManager(OUTPUTS)
_DEVICE = perceive.pick_device()

DEFAULT_SETTINGS = {"strength": 0.6, "background": 0.15, "glow": 0.35, "smoothing": 0.6,
                    "focus": "", "use_claude": True}


def _settings(raw: str | dict | None) -> dict:
    s = dict(DEFAULT_SETTINGS)
    if isinstance(raw, str) and raw.strip():
        raw = json.loads(raw)
    if isinstance(raw, dict):
        s.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
    return s


@app.get("/api/health")
def health():
    return {"ok": True, "device": _DEVICE, "claude_available": understand.available(),
            "claude_model": understand.DEFAULT_MODEL, "defaults": DEFAULT_SETTINGS}


@app.get("/api/samples")
def samples():
    if not SAMPLES.exists():
        return []
    return [{"name": p.name, "size": p.stat().st_size}
            for p in sorted(SAMPLES.iterdir()) if p.suffix.lower() in VIDEO_EXT]


@app.get("/api/jobs")
def list_jobs():
    return sorted((j.public() for j in jobs.jobs.values()), key=lambda j: -j["created"])


@app.post("/api/process")
async def process(file: UploadFile | None = File(None), sample: str | None = Form(None),
                  settings: str | None = Form(None)):
    s = _settings(settings)
    if sample:
        src = (SAMPLES / sample).resolve()
        if src.parent != SAMPLES.resolve() or not src.exists():
            raise HTTPException(404, "sample not found")
        return jobs.submit_file(src, sample, s).public()
    if file is None:
        raise HTTPException(400, "send a file or a sample name")
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, f"unsupported file type {suffix}")
    fd, name = tempfile.mkstemp(suffix=suffix, dir=OUTPUTS)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(file.file, f, 8 << 20)
        if tmp.stat().st_size == 0:
            raise HTTPException(400, "The file is empty.")
        return jobs.submit_file(tmp, file.filename or tmp.name, s, move=True).public()
    finally:
        tmp.unlink(missing_ok=True)  # moved into the job dir on success; removed on failure


class PathRequest(BaseModel):
    input_path: str
    settings: dict | None = None


@app.post("/api/process_path")
def process_path(req: PathRequest):
    src = Path(req.input_path).expanduser()
    if not src.exists():
        raise HTTPException(404, "file not found")
    return jobs.submit_file(src, src.name, _settings(req.settings)).public()


class RerenderRequest(BaseModel):
    settings: dict


@app.post("/api/rerender/{jid}")
def rerender(jid: str, req: RerenderRequest):
    if jid not in jobs.jobs:
        raise HTTPException(404, "job not found")
    try:
        return jobs.submit_rerender(jid, _settings({**jobs.jobs[jid].settings, **req.settings})).public()
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/status/{jid}")
def status(jid: str):
    j = jobs.jobs.get(jid)
    if j is None:
        raise HTTPException(404, "job not found")
    return j.public()


@app.get("/api/result/{jid}")
def result(jid: str):
    j = jobs.jobs.get(jid)
    if j is None:
        raise HTTPException(404, "job not found")
    if not (j.dir / "metadata.json").exists():
        raise HTTPException(409, "not ready")
    v = j.version
    base = f"/api/files/{jid}"
    return {**j.public(), "original": f"{base}/original.mp4?v={v}", "enhanced": f"{base}/enhanced.mp4?v={v}",
            "attention": f"{base}/attention.mp4?v={v}",
            "metadata": json.loads((j.dir / "metadata.json").read_text())}


@app.api_route("/api/files/{jid}/{name}", methods=["GET", "HEAD"])
def files(jid: str, name: str):
    j = jobs.jobs.get(jid)
    if j is None or name not in SERVABLE or not (j.dir / name).exists():
        raise HTTPException(404, "not found")
    # FileResponse honours HTTP Range requests, which the <video> element needs to seek.
    return FileResponse(j.dir / name, headers={"Cache-Control": "no-cache"})


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))

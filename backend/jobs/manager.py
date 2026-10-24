"""In-memory job manager with one worker thread. No database, no queue service.

Finished jobs are also rediscovered from disk at startup, so pre-rendered demo
results survive a backend restart.
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pipeline import run as pipeline


@dataclass
class Job:
    id: str
    dir: Path
    name: str
    settings: dict
    status: str = "queued"  # queued | processing | complete | error
    stage: str = "queued"
    label: str = "Waiting"
    progress: float = 0.0
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    version: int = 0  # bumps on each successful render (cache-busting for the player)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "stage": self.stage,
                "label": self.label, "progress": round(self.progress, 4), "error": self.error,
                "settings": self.settings, "version": self.version, "created": self.created}


class JobManager:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self._discover()
        threading.Thread(target=self._worker, daemon=True).start()

    def _discover(self) -> None:
        for d in sorted(self.root.iterdir()):
            meta, info = d / "metadata.json", d / "job.json"
            if meta.exists() and (d / "enhanced.mp4").exists():
                extra = json.loads(info.read_text()) if info.exists() else {}
                j = Job(d.name, d, extra.get("name", d.name), extra.get("settings", {}),
                        status="complete", stage="done", label="Done", progress=1.0,
                        created=extra.get("created", d.stat().st_mtime), version=1)
                self.jobs[j.id] = j

    def _save_info(self, j: Job) -> None:
        (j.dir / "job.json").write_text(json.dumps({"name": j.name, "settings": j.settings, "created": j.created}))

    # ---- submission -----------------------------------------------------------
    def submit_file(self, src: Path, name: str, settings: dict, move: bool = False) -> Job:
        jid = uuid.uuid4().hex[:10]
        d = self.root / jid
        d.mkdir(parents=True)
        dst = d / f"input{src.suffix.lower() or '.mp4'}"
        (shutil.move if move else shutil.copyfile)(str(src), str(dst))
        j = Job(jid, d, name, settings)
        self._save_info(j)
        with self.lock:
            self.jobs[jid] = j
        self.q.put(("process", jid, dst))
        return j

    def submit_rerender(self, jid: str, settings: dict) -> Job:
        j = self.jobs[jid]
        if j.status in ("queued", "processing"):
            raise RuntimeError("job is busy")
        j.settings = {**j.settings, **settings}
        self._save_info(j)
        j.status, j.stage, j.label, j.progress, j.error = "queued", "queued", "Waiting", 0.0, None
        self.q.put(("rerender", jid, None))
        return j

    # ---- worker -------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            kind, jid, src = self.q.get()
            j = self.jobs.get(jid)
            if j is None:
                continue
            j.status = "processing"

            def cb(stage, frac, label, j=j):
                j.stage, j.progress, j.label, j.updated = stage, frac, label, time.time()

            try:
                if kind == "process":
                    pipeline.process(j.dir, src, j.settings, cb)
                else:
                    pipeline.rerender(j.dir, j.settings, cb)
                j.status, j.stage, j.label, j.progress = "complete", "done", "Done", 1.0
                j.version += 1
            except Exception as e:
                traceback.print_exc()
                # ValueErrors from the pipeline are written for people; others aren't.
                msg = str(e) if isinstance(e, ValueError) else f"Something went wrong while processing ({type(e).__name__}: {e})"
                j.status, j.error = "error", msg[:500]
                for leftover in j.dir.glob("*.part"):
                    leftover.unlink(missing_ok=True)
            j.updated = time.time()

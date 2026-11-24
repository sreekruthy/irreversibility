"""Resume-or-restart policy shared by extract.py, stated.py and agent.py.

Each output gets a status file <out>.status.json:
  {"state": "running" | "done" | "error", "config": {...}, "config_hash": ..., ...}

On start:
  done     + same config  -> nothing to do (prints ALREADY COMPLETE and exits)
  running  + same config  -> the previous run was cut off (timeout, session end, kill):
                             RESUME from the completed items
  error                   -> the previous run crashed: its partial outputs are moved to
                             <out>.failed-<UTC time> (kept for the record, never deleted)
                             and the run STARTS FROM SCRATCH
  different config        -> same as error: archive and start from scratch, so outputs
                             from different settings are never mixed
  no status file          -> fresh start (old outputs without a status are archived)

Any exception inside the guarded block marks the status "error" (with the traceback)
and re-raises, so the next start knows to restart from scratch.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback


def _now():
    return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


class RunGuard:
    def __init__(self, out: str, config: dict, related: tuple[str, ...] = ()):
        """out: output file or directory. related: extra files written next to it."""
        self.out = out.rstrip("/")
        self.status_path = self.out + ".status.json"
        self.config = config
        self.hash = config_hash(config)
        self.related = related
        self.mode = None

    def _paths(self):
        return [p for p in (self.out, *self.related) if os.path.exists(p)]

    def _archive(self, reason):
        stamp = self.out + f".failed-{_now()}"
        os.makedirs(stamp, exist_ok=True)
        for p in self._paths() + ([self.status_path] if os.path.exists(self.status_path) else []):
            shutil.move(p, os.path.join(stamp, os.path.basename(p)))
        with open(os.path.join(stamp, "REASON.txt"), "w") as f:
            f.write(reason + "\n")
        print(f"[runguard] previous outputs archived to {stamp} ({reason}); starting from scratch", flush=True)

    def _write(self, state, **extra):
        os.makedirs(os.path.dirname(self.status_path) or ".", exist_ok=True)
        rec = {"state": state, "config": self.config, "config_hash": self.hash, "updated": _now(), **extra}
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(tmp, self.status_path)

    def __enter__(self):
        st = None
        if os.path.exists(self.status_path):
            try:
                st = json.load(open(self.status_path))
            except Exception:
                st = {"state": "error", "config_hash": None}
        if st is None:
            if self._paths():
                self._archive("outputs existed without a status file")
            self.mode = "fresh"
        elif st.get("config_hash") != self.hash:
            self._archive("settings changed since the previous run")
            self.mode = "fresh"
        elif st["state"] == "error":
            self._archive("previous run ended with an error")
            self.mode = "fresh"
        elif st["state"] == "done":
            print(f"[runguard] ALREADY COMPLETE: {self.out} (nothing to do)", flush=True)
            self.mode = "done"
            return self
        else:
            print(f"[runguard] previous run was interrupted: RESUMING {self.out}", flush=True)
            self.mode = "resume"
        self._write("running", started=_now(), mode=self.mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.mode == "done":
            return False
        if exc_type is None:
            self._write("done", finished=_now())
            return False
        if exc_type in (KeyboardInterrupt, SystemExit):
            # stopped by the user or the platform: leave "running" so the next start resumes
            print("[runguard] interrupted; the next start will resume", flush=True)
            return False
        self._write("error", error="".join(traceback.format_exception(exc_type, exc, tb))[-4000:])
        print("[runguard] run FAILED; the next start will begin from scratch", file=sys.stderr, flush=True)
        return False


def read_jsonl_tolerant(path):
    """Read a JSONL file, dropping a truncated final line left by a killed run."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        lines = f.read().split("\n")
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i >= len(lines) - 2:          # only the last line may be partial
                print(f"[runguard] dropped a truncated last line in {path}", flush=True)
                with open(path, "w") as g:   # rewrite without it so later appends stay valid
                    for r in rows:
                        g.write(json.dumps(r, ensure_ascii=False) + "\n")
                break
            raise
    return rows

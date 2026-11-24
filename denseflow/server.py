#!/usr/bin/env python3
"""Zero-dependency HTTP API + static player server.

Routes
------
GET  /                         web/index.html
GET  /web/<file>               player assets
GET  /artifacts/<path>         experiment outputs (PNG/JSON/CSV)
GET  /api/health               liveness
GET  /api/sequences            experiments/output/manifest.json
GET  /api/pair?seq=&pair=      one precomputed pair JSON
GET  /api/rebranch?seq=&pair=&index=  re-run a pair and return one branch view
POST /api/analyze              multipart: frame_a, frame_b, optional mask,
                               and a JSON ``params`` form field
POST /api/branch?job_id=&index=       switch branch inside an upload job

Start:  python3 -m denseflow.server   (PORT env optional, default 8000)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from denseflow.pipeline import analyze_pair, api_payload, branch_payload  # noqa: E402
from denseflow.runner import rebuild_pair_result  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")
ART_DIR = os.path.join(ROOT, "experiments", "output")

JOBS: dict[str, dict] = {}
REBUILD_CACHE: dict[str, dict] = {}
MAX_JOBS = 12

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".csv": "text/csv; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


# ---------------------------------------------------------------------------
# Minimal multipart/form-data parser (no cgi dependency).
# ---------------------------------------------------------------------------

def parse_multipart(body: bytes, content_type: str) -> dict:
    marker = "boundary="
    i = content_type.find(marker)
    if i < 0:
        return {}
    boundary = content_type[i + len(marker):].strip().strip('"')
    delim = ("--" + boundary).encode()
    parts = body.split(delim)
    out: dict[str, dict] = {}
    for part in parts:
        if not part or part in (b"--", b"--\r\n", b"\r\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not _:
            continue
        content = content.rstrip(b"\r\n")
        headers = header_blob.decode("utf-8", "replace")
        name = None
        filename = None
        for token in headers.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"')
            elif token.startswith("filename="):
                filename = token[9:].strip('"')
        if name:
            out[name] = {"filename": filename, "content": content,
                         "headers": headers}
    return out


def _part_image(part: dict | None) -> np.ndarray | None:
    if part is None or not part.get("content"):
        return None
    im = Image.open(io.BytesIO(part["content"])).convert("RGB")
    return np.array(im)


def _mask_from_part(part: dict | None, shape) -> np.ndarray | None:
    if part is None or not part.get("content"):
        return None
    im = Image.open(io.BytesIO(part["content"]))
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        arr = np.array(rgba)
        m = (arr[..., 3] > 20) | (arr[..., :3].mean(axis=2) > 40)
    else:
        g = np.array(im.convert("L"))
        m = g > 40
    if m.shape != shape[:2]:
        im2 = Image.fromarray((m.astype(np.uint8) * 255)).resize(
            (shape[1], shape[0]), Image.NEAREST)
        m = np.array(im2) > 0
    return m


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "DenseFlow/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("[http] %s\n" % (fmt % args))

    # ---- helpers ---------------------------------------------------------
    def _json(self, obj, status: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: str):
        if not os.path.isfile(path):
            self._json({"error": "not found", "path": path}, 404)
            return
        ext = os.path.splitext(path)[1].lower()
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(path, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if ext in (".html", ".js", ".css"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _safe_artifact(self, rel: str) -> str | None:
        base = os.path.realpath(ART_DIR)
        target = os.path.realpath(os.path.join(base, rel))
        if target != base and not target.startswith(base + os.sep):
            return None
        return target

    # ---- GET -------------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/health":
            return self._json({"ok": True, "time": time.time(),
                               "artifacts": os.path.isdir(ART_DIR)})

        if path == "/api/sequences":
            mf = os.path.join(ART_DIR, "manifest.json")
            if not os.path.isfile(mf):
                return self._json({
                    "sequences": [],
                    "warning": "experiments/output/manifest.json missing; "
                               "run `python3 run_experiments.py` first.",
                })
            with open(mf) as fh:
                return self._json(json.load(fh))

        if path == "/api/pair":
            seq = (qs.get("seq") or [""])[0]
            pair = int((qs.get("pair") or ["0"])[0])
            pj = os.path.join(ART_DIR, seq, f"pair_{pair:03d}.json")
            target = self._safe_artifact(os.path.join(seq, f"pair_{pair:03d}.json"))
            if target is None or not os.path.isfile(pj):
                return self._json({"error": "pair not found",
                                   "seq": seq, "pair": pair}, 404)
            with open(pj) as fh:
                return self._json(json.load(fh))

        if path == "/api/rebranch":
            return self.handle_rebranch(qs)

        if path.startswith("/artifacts/"):
            rel = path[len("/artifacts/"):]
            target = self._safe_artifact(rel)
            if target is None:
                return self._json({"error": "bad path"}, 400)
            return self._file(target)

        if path.startswith("/web/"):
            rel = path[len("/web/"):]
            target = os.path.realpath(os.path.join(WEB_DIR, rel))
            if not target.startswith(WEB_DIR + os.sep):
                return self._json({"error": "bad path"}, 400)
            return self._file(target)

        if path in ("/", "/index.html"):
            return self._file(os.path.join(WEB_DIR, "index.html"))

        self._json({"error": "unknown route", "path": path}, 404)

    def handle_rebranch(self, qs):
        seq = (qs.get("seq") or [""])[0]
        pair = int((qs.get("pair") or ["0"])[0])
        index = int((qs.get("index") or ["0"])[0])
        pj_path = os.path.join(ART_DIR, seq, f"pair_{pair:03d}.json")
        if not os.path.isfile(pj_path):
            return self._json({"error": "pair json missing; run experiments"}, 404)
        try:
            with open(pj_path) as fh:
                pair_json = json.load(fh)
            key = f"{seq}:{pair}"
            if key not in REBUILD_CACHE:
                REBUILD_CACHE[key] = rebuild_pair_result(seq, pair, pair_json)
            res = REBUILD_CACHE[key]
            if not (0 <= index < len(res["branches"])):
                return self._json({"error": "bad branch index",
                                   "n": len(res["branches"])}, 400)
            return self._json(branch_payload(res, index))
        except Exception as exc:  # pragma: no cover
            import traceback
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)

    # ---- POST ------------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/analyze":
            return self.handle_analyze()
        if parsed.path == "/api/branch":
            qs = parse_qs(parsed.query)
            job_id = (qs.get("job_id") or [""])[0]
            index = int((qs.get("index") or ["0"])[0])
            job = JOBS.get(job_id)
            if job is None:
                return self._json({"error": "unknown job_id"}, 404)
            res = job["res"]
            if not (0 <= index < len(res.get("branches", []))):
                return self._json({"error": "bad branch index"}, 400)
            try:
                return self._json(branch_payload(res, index))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return self._json({"error": str(exc)}, 500)
        self._json({"error": "unknown route"}, 404)

    def handle_analyze(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in ctype:
                return self._json({"error": "expected multipart/form-data"}, 400)
            parts = parse_multipart(body, ctype)
            img_a = _part_image(parts.get("frame_a"))
            img_b = _part_image(parts.get("frame_b"))
            if img_a is None or img_b is None:
                return self._json({"error": "frame_a and frame_b required"}, 400)
            if img_a.shape != img_b.shape:
                h = min(img_a.shape[0], img_b.shape[0])
                w = min(img_a.shape[1], img_b.shape[1])
                img_a = img_a[:h, :w]
                img_b = img_b[:h, :w]

            params = {}
            if "params" in parts:
                raw = parts["params"]["content"].decode("utf-8", "replace")
                try:
                    params = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    params = {}
            mask = _mask_from_part(parts.get("mask"), img_a.shape)

            t0 = time.time()
            res = analyze_pair(
                img_a, img_b, params,
                mask_a=mask,
                run_candidates=bool(params.get("run_candidates", True)),
            )
            payload = api_payload(res)
            job_id = uuid.uuid4().hex[:12]
            JOBS[job_id] = {"res": res, "created": time.time()}
            if len(JOBS) > MAX_JOBS:
                oldest = sorted(JOBS, key=lambda k: JOBS[k]["created"])[:-MAX_JOBS]
                for k in oldest:
                    JOBS.pop(k, None)
            payload["job_id"] = job_id
            payload["compute_sec"] = round(time.time() - t0, 2)
            return self._json(payload)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)


def main():
    port = int(os.environ.get("PORT", "8000"))
    os.makedirs(WEB_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"DenseFlow player:  http://localhost:{port}/")
    print(f"web assets:        {WEB_DIR}")
    print(f"artifacts:         {ART_DIR}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

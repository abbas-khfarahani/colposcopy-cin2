#!/usr/bin/env python3
"""Local web UI: scrollable gallery of ALL Orginal Data images + delete.

Usage:
  python tools/original_data_cleanup/review_all_images.py
  open http://127.0.0.1:8766
"""

from __future__ import annotations

import io
import json
import re
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file
from PIL import Image

import sys

_KIT = Path(__file__).resolve().parent
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))
from _paths import ROOT, DATA, LOGS, CACHE, ensure_dirs

LOG_PATH = LOGS / "all_review_log.jsonl"
CACHE_DIR = CACHE / "all_preview"

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MOD_RE = re.compile(r"_([AGLDZ])(\d*)\.(?:jpe?g|png|tiff?|bmp)$", re.I)

app = Flask(__name__)
_files: list[str] = []
_lock = threading.Lock()


def scan_images() -> list[str]:
    files = []
    for f in DATA.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXT:
            files.append(str(f))
    files.sort(key=str.lower)
    return files


def refresh_index() -> None:
    global _files
    with _lock:
        _files = scan_images()


def get_files() -> list[str]:
    with _lock:
        return list(_files)


def log_event(event: dict) -> None:
    event = {**event, "ts": datetime.now().isoformat(timespec="seconds")}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def case_id_of(path: Path) -> str:
    try:
        return path.relative_to(DATA).parts[0]
    except Exception:
        return path.parent.name


def modality_of(name: str) -> str:
    m = MOD_RE.search(name)
    if m:
        return f"{m.group(1).upper()}{m.group(2) or ''}"
    return "—"


def preview_bytes(src: Path, max_side: int, quality: int = 72) -> bytes:
    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        st = src.stat()
        mtime, size = int(st.st_mtime), st.st_size
    except OSError:
        abort(404)
    safe = re.sub(r"[^\w.\-]+", "_", f"{mtime}_{size}_{max_side}_{src.name}")[:180]
    cache_path = CACHE_DIR / f"{safe}.jpg"
    if cache_path.exists():
        return cache_path.read_bytes()
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
    try:
        cache_path.write_bytes(data)
    except OSError:
        pass
    return data


PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>All images gallery</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --fg:#e7ecf3; --muted:#9aa7b8; --acc:#3d8bfd; --bad:#c92a2a; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--fg); }
    header {
      position: sticky; top:0; z-index:20;
      padding:12px 16px; background:rgba(15,20,25,.94); backdrop-filter: blur(8px);
      border-bottom:1px solid #2a3545; display:flex; gap:12px; align-items:center; flex-wrap:wrap;
    }
    header .meta { color:var(--muted); font-size:13px; }
    .grid {
      display:grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap:12px;
      padding:14px 16px 40px;
    }
    .card {
      background:var(--card); border-radius:12px; overflow:hidden;
      display:flex; flex-direction:column; border:1px solid #243041;
    }
    .card.gone { opacity:.35; filter:grayscale(1); pointer-events:none; }
    .card img {
      width:100%; height:200px; object-fit:contain; background:#000; display:block; cursor:zoom-in;
    }
    .info { padding:8px 10px 10px; }
    .path { font-family: ui-monospace, Consolas, monospace; font-size:11px; color:var(--muted); word-break:break-all; min-height:2.6em; }
    .row { display:flex; gap:8px; align-items:center; margin-top:8px; }
    .badge { background:#2a3545; border-radius:999px; padding:3px 8px; font-size:11px; color:var(--muted); }
    button.del {
      margin-left:auto; border:0; border-radius:8px; padding:8px 12px;
      background:var(--bad); color:#fff; font-weight:700; cursor:pointer;
    }
    button.del:hover { filter:brightness(1.08); }
    /* lightbox */
    #lb {
      display:none; position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.92);
      align-items:center; justify-content:center; padding:20px;
    }
    #lb.open { display:flex; }
    #lb img { max-width:96vw; max-height:92vh; object-fit:contain; border-radius:8px; }
    #lb .close {
      position:absolute; top:12px; right:16px; border:0; background:#333; color:#fff;
      border-radius:8px; padding:8px 12px; font-weight:700; cursor:pointer;
    }
  </style>
</head>
<body>
  <header>
    <strong>All images — scroll &amp; delete</strong>
    <span class="meta" id="count">{{ total }} images</span>
    <span class="meta">deleted: <b id="deleted">0</b></span>
    <span class="meta">Click image = big view · Delete removes file</span>
  </header>

  <div class="grid" id="grid">
    {% for item in items %}
    <div class="card" id="card-{{ item.i }}" data-i="{{ item.i }}">
      <img loading="lazy" decoding="async"
           src="/img?i={{ item.i }}&size=420"
           data-full="/img?i={{ item.i }}&size=1400"
           alt="{{ item.name }}"
           onclick="openLb(this)"/>
      <div class="info">
        <div class="path">{{ item.rel }}</div>
        <div class="row">
          <span class="badge">{{ item.case }}</span>
          <span class="badge">{{ item.mod }}</span>
          <button class="del" type="button" onclick="delImg({{ item.i }}, this)">Delete</button>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>

  <div id="lb" onclick="if(event.target===this) closeLb()">
    <button class="close" type="button" onclick="closeLb()">Close ✕</button>
    <img id="lbimg" alt="full"/>
  </div>

<script>
let deleted = 0;
function openLb(img) {
  const lb = document.getElementById('lb');
  const lbimg = document.getElementById('lbimg');
  lbimg.src = img.dataset.full || img.src;
  lb.classList.add('open');
}
function closeLb() {
  document.getElementById('lb').classList.remove('open');
  document.getElementById('lbimg').src = '';
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLb();
});

async function delImg(i, btn) {
  const card = document.getElementById('card-' + i);
  const path = card?.querySelector('.path')?.textContent || ('#' + i);
  if (!confirm('Delete permanently?\n' + path)) return;
  btn.disabled = true;
  try {
    const res = await fetch('/delete', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({i})
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      alert('Delete failed: ' + (data.error || res.status));
      btn.disabled = false;
      return;
    }
    card.classList.add('gone');
    deleted += 1;
    document.getElementById('deleted').textContent = deleted;
  } catch (err) {
    alert('Delete failed: ' + err);
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


@app.get("/")
def index():
    files = get_files()
    items = []
    for i, s in enumerate(files):
        p = Path(s)
        try:
            rel = str(p.relative_to(DATA))
        except Exception:
            rel = p.name
        items.append(
            {
                "i": i,
                "rel": rel,
                "name": p.name,
                "case": case_id_of(p),
                "mod": modality_of(p.name),
            }
        )
    return render_template_string(PAGE, items=items, total=len(items))


@app.get("/img")
def image():
    files = get_files()
    i = int(request.args.get("i", 0))
    size = int(request.args.get("size", 420))
    size = max(200, min(1600, size))
    if i < 0 or i >= len(files):
        abort(404)
    path = Path(files[i])
    if not path.exists():
        abort(404)
    quality = 70 if size <= 500 else 80
    data = preview_bytes(path, max_side=size, quality=quality)
    return send_file(io.BytesIO(data), mimetype="image/jpeg", max_age=3600)


@app.post("/delete")
def delete():
    payload = request.get_json(silent=True) or {}
    try:
        i = int(payload.get("i"))
    except Exception:
        return jsonify(ok=False, error="bad index"), 400
    files = get_files()
    if i < 0 or i >= len(files):
        return jsonify(ok=False, error="out of range"), 404
    path = Path(files[i])
    if not path.exists():
        # already gone — mark slot empty-ish
        return jsonify(ok=True, already=True)
    try:
        rel = str(path.relative_to(DATA))
        path.unlink()
        log_event({"action": "DELETE", "from": rel})
        # keep index positions stable: leave a tombstone empty string
        with _lock:
            if i < len(_files) and _files[i] == str(path):
                _files[i] = ""
        return jsonify(ok=True, from_path=rel)
    except Exception as e:
        log_event({"action": "ERROR", "error": str(e)})
        return jsonify(ok=False, error=str(e)), 500


def main():
    ensure_dirs()
    if not DATA.is_dir():
        raise SystemExit(f"Data folder not found: {DATA}")
    print("Scanning all images (one-time)...")
    refresh_index()
    print(f"Images found: {len(get_files())}")
    print("Open: http://127.0.0.1:8766")
    app.run(host="127.0.0.1", port=8766, debug=False, threaded=True)


if __name__ == "__main__":
    main()

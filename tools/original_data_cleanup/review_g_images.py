#!/usr/bin/env python3
"""Scrollable gallery of _G images. Leave OK ones alone; rename/delete if wrong.

Usage:
  python tools/original_data_cleanup/review_g_images.py
  open http://127.0.0.1:8769
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

LOG_PATH = LOGS / "g_review_log.jsonl"
CACHE_DIR = CACHE / "g_preview"

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
G_RE = re.compile(r"_G\d*\.(?:jpe?g|png|tiff?|bmp)$", re.I)
MOD_RE = re.compile(r"_([AGLDZ])(\d*)\.(?:jpe?g|png|tiff?|bmp)$", re.I)
PREFIX_RE = re.compile(r"^(.*)_G\d*$", re.I)

app = Flask(__name__)
_files: list[str] = []
_lock = threading.Lock()


def scan_g_images() -> list[str]:
    files = []
    for f in DATA.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXT and G_RE.search(f.name):
            files.append(str(f))
    files.sort(key=str.lower)
    return files


def refresh_index() -> None:
    global _files
    with _lock:
        _files = scan_g_images()


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


def sibling_one(src: Path, letter: str) -> Path | None:
    try:
        kids = sorted(src.parent.iterdir())
    except OSError:
        return None
    for f in kids:
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        if f.name == src.name:
            continue
        m = MOD_RE.search(f.name)
        if m and m.group(1).upper() == letter.upper():
            return f
    return None


def next_name(src: Path, letter: str) -> Path:
    m = PREFIX_RE.match(src.stem)
    if not m:
        raise ValueError(f"Cannot parse prefix from {src.name}")
    prefix = m.group(1)
    ext = src.suffix
    used: set[int] = set()
    for f in src.parent.iterdir():
        if not f.is_file():
            continue
        mm = MOD_RE.search(f.name)
        if mm and mm.group(1).upper() == letter.upper():
            num = mm.group(2)
            used.add(int(num) if num else 1)
    n = 1
    while n in used:
        n += 1
    return src.parent / f"{prefix}_{letter.upper()}{n}{ext}"


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
  <title>G images — scroll review</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --fg:#e7ecf3; --muted:#9aa7b8; --bad:#c92a2a; }
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
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap:14px;
      padding:14px 16px 48px;
    }
    .card {
      background:var(--card); border-radius:12px; overflow:hidden;
      border:1px solid #243041; display:flex; flex-direction:column;
    }
    .card.gone { opacity:.3; filter:grayscale(1); pointer-events:none; }
    .card img.main {
      width:100%; height:220px; object-fit:contain; background:#000; display:block; cursor:zoom-in;
    }
    .info { padding:8px 10px 10px; }
    .path { font-family: ui-monospace, Consolas, monospace; font-size:11px; color:var(--muted); word-break:break-all; min-height:2.4em; }
    .badge { background:#2a3545; border-radius:999px; padding:3px 8px; font-size:11px; color:var(--muted); margin-right:4px; }
    .compare {
      display:grid; grid-template-columns: repeat(3, 1fr); gap:4px; padding:0 8px 8px;
    }
    .compare figure { margin:0; background:#0b1016; border-radius:6px; padding:4px; text-align:center; }
    .compare img { width:100%; height:64px; object-fit:contain; background:#000; border-radius:4px; display:block; }
    .compare figcaption { font-size:10px; color:var(--muted); margin-top:2px; }
    .compare .none { height:64px; display:flex; align-items:center; justify-content:center; color:#556; font-size:10px; }
    .actions { display:flex; flex-wrap:wrap; gap:6px; padding:0 10px 10px; }
    .actions button {
      border:0; border-radius:8px; padding:8px 10px; font-size:13px; font-weight:700; cursor:pointer; color:#061018;
    }
    .A { background:#ff6b6b; color:#fff; } .L { background:#fcc419; } .Z { background:#74c0fc; }
    .del { background:var(--bad); color:#fff; margin-left:auto; }
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
    <strong>Review _G images — scroll</strong>
    <span class="meta">{{ total }} images</span>
    <span class="meta">changed: <b id="changed">0</b></span>
    <span class="meta">If OK → do nothing. If wrong → →A / →L / →Z or Delete</span>
  </header>

  <div class="grid">
    {% for item in items %}
    <div class="card" id="card-{{ item.i }}" data-i="{{ item.i }}">
      <img class="main" loading="lazy" decoding="async"
           src="/img?i={{ item.i }}&size=480"
           data-full="/img?i={{ item.i }}&size=1400"
           alt="{{ item.name }}"
           onclick="openLb(this)"/>
      <div class="info">
        <div class="path">{{ item.rel }}</div>
        <span class="badge">{{ item.case }}</span>
        <span class="badge">{{ item.mod }}</span>
      </div>
      <div class="compare">
        <figure>
          {% if item.has_a %}
          <img loading="lazy" src="/img?i={{ item.i }}&kind=ref&letter=A&size=200" alt="A"/>
          <figcaption>A</figcaption>
          {% else %}
          <div class="none">no A</div><figcaption>A</figcaption>
          {% endif %}
        </figure>
        <figure>
          {% if item.has_l %}
          <img loading="lazy" src="/img?i={{ item.i }}&kind=ref&letter=L&size=200" alt="L"/>
          <figcaption>L</figcaption>
          {% else %}
          <div class="none">no L</div><figcaption>L</figcaption>
          {% endif %}
        </figure>
        <figure>
          {% if item.has_z %}
          <img loading="lazy" src="/img?i={{ item.i }}&kind=ref&letter=Z&size=200" alt="Z"/>
          <figcaption>Z</figcaption>
          {% else %}
          <div class="none">no Z</div><figcaption>Z</figcaption>
          {% endif %}
        </figure>
      </div>
      <div class="actions">
        <button class="A" type="button" onclick="act({{ item.i }}, 'A', this)">→ A</button>
        <button class="L" type="button" onclick="act({{ item.i }}, 'L', this)">→ L</button>
        <button class="Z" type="button" onclick="act({{ item.i }}, 'Z', this)">→ Z</button>
        <button class="del" type="button" onclick="act({{ item.i }}, 'DELETE', this)">Delete</button>
      </div>
    </div>
    {% endfor %}
  </div>

  <div id="lb" onclick="if(event.target===this) closeLb()">
    <button class="close" type="button" onclick="closeLb()">Close ✕</button>
    <img id="lbimg" alt="full"/>
  </div>

<script>
let changed = 0;
function openLb(img) {
  document.getElementById('lbimg').src = img.dataset.full || img.src;
  document.getElementById('lb').classList.add('open');
}
function closeLb() {
  document.getElementById('lb').classList.remove('open');
  document.getElementById('lbimg').src = '';
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLb(); });

async function act(i, action, btn) {
  const card = document.getElementById('card-' + i);
  const path = card?.querySelector('.path')?.textContent || ('#' + i);
  const msg = action === 'DELETE'
    ? ('Delete permanently?\n' + path)
    : ('Rename this _G to ' + action + '?\n' + path);
  if (!confirm(msg)) return;
  const buttons = card.querySelectorAll('button');
  buttons.forEach(b => b.disabled = true);
  try {
    const res = await fetch('/act', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({i, action})
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      alert('Failed: ' + (data.error || res.status));
      buttons.forEach(b => b.disabled = false);
      return;
    }
    card.classList.add('gone');
    changed += 1;
    document.getElementById('changed').textContent = changed;
  } catch (err) {
    alert('Failed: ' + err);
    buttons.forEach(b => b.disabled = false);
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
        m = MOD_RE.search(p.name)
        mod = f"{m.group(1).upper()}{m.group(2) or ''}" if m else "G"
        items.append(
            {
                "i": i,
                "rel": rel,
                "name": p.name,
                "case": case_id_of(p),
                "mod": mod,
                "has_a": sibling_one(p, "A") is not None,
                "has_l": sibling_one(p, "L") is not None,
                "has_z": sibling_one(p, "Z") is not None,
            }
        )
    return render_template_string(PAGE, items=items, total=len(items))


@app.get("/img")
def image():
    files = get_files()
    i = int(request.args.get("i", 0))
    size = int(request.args.get("size", 480))
    size = max(120, min(1600, size))
    if i < 0 or i >= len(files):
        abort(404)
    path = Path(files[i])
    if not path or not str(path) or not path.exists():
        abort(404)
    kind = request.args.get("kind", "g")
    if kind == "ref":
        letter = request.args.get("letter", "A").upper()
        ref = sibling_one(path, letter)
        if not ref:
            abort(404)
        path = ref
    quality = 70 if size <= 500 else 80
    data = preview_bytes(path, max_side=size, quality=quality)
    return send_file(io.BytesIO(data), mimetype="image/jpeg", max_age=3600)


@app.post("/act")
def act():
    payload = request.get_json(silent=True) or {}
    try:
        i = int(payload.get("i"))
    except Exception:
        return jsonify(ok=False, error="bad index"), 400
    action = str(payload.get("action", "")).upper()
    files = get_files()
    if i < 0 or i >= len(files) or not files[i]:
        return jsonify(ok=False, error="out of range"), 404
    path = Path(files[i])
    if not path.exists():
        return jsonify(ok=True, already=True)

    try:
        rel = str(path.relative_to(DATA))
    except Exception:
        rel = path.name

    try:
        if action == "DELETE":
            path.unlink()
            log_event({"action": "DELETE", "from": rel})
        elif action in {"A", "L", "Z"}:
            dest = next_name(path, action)
            if dest.exists():
                return jsonify(ok=False, error=f"exists: {dest.name}"), 409
            path.rename(dest)
            log_event({"action": action, "from": rel, "to": str(dest.relative_to(DATA))})
        else:
            return jsonify(ok=False, error="bad action"), 400

        with _lock:
            if i < len(_files) and _files[i] == str(path):
                _files[i] = ""
        return jsonify(ok=True)
    except Exception as e:
        log_event({"action": "ERROR", "from": rel, "error": str(e)})
        return jsonify(ok=False, error=str(e)), 500


def main():
    ensure_dirs()
    if not DATA.is_dir():
        raise SystemExit(f"Data folder not found: {DATA}")
    print("Scanning _G images...")
    refresh_index()
    print(f"_G images: {len(get_files())}")
    print("Open: http://127.0.0.1:8769")
    app.run(host="127.0.0.1", port=8769, debug=False, threaded=True)


if __name__ == "__main__":
    main()

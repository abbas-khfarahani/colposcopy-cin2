#!/usr/bin/env python3
"""Fast local web UI to reclassify _D images as A / G / L / Z.

Optimizations:
  - scan Orginal Data once, keep list in memory
  - serve JPEG previews (not full 6–8MB originals)
  - disk thumbnail cache
  - only first ref image per modality

Usage:
  python tools/original_data_cleanup/review_d_images.py
  open http://127.0.0.1:8765
"""

from __future__ import annotations

import io
import json
import re
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for
from PIL import Image

import sys

_KIT = Path(__file__).resolve().parent
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))
from _paths import ROOT, DATA, LOGS, CACHE, ensure_dirs

LOG_PATH = LOGS / "d_review_log.jsonl"
STATE_PATH = LOGS / "d_review_state.json"
CACHE_DIR = CACHE / "d_preview"

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
D_RE = re.compile(r"_D\d*\.(?:jpe?g|png|tiff?|bmp)$", re.I)
MOD_RE = re.compile(r"_([AGLDZ])(\d*)\.(?:jpe?g|png|tiff?|bmp)$", re.I)
PREFIX_RE = re.compile(r"^(.*)_D\d*$", re.I)

app = Flask(__name__)

# In-memory index (absolute paths as strings for stability)
_files: list[str] = []
_lock = threading.Lock()


def scan_d_images() -> list[str]:
    files = []
    for f in DATA.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMG_EXT and D_RE.search(f.name):
            files.append(str(f))
    files.sort(key=str.lower)
    return files


def refresh_index() -> None:
    global _files
    with _lock:
        _files = scan_d_images()


def get_files() -> list[str]:
    with _lock:
        return list(_files)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"index": 0, "done": []}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def log_event(event: dict) -> None:
    event = {**event, "ts": datetime.now().isoformat(timespec="seconds")}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def case_id_of(path: Path) -> str:
    try:
        return path.relative_to(DATA).parts[0]
    except Exception:
        return path.parent.name


def sibling_mods(d_path: Path) -> dict[str, list[Path]]:
    """Other modality images in the same folder (max 1 per letter for speed)."""
    out: dict[str, list[Path]] = {k: [] for k in "AGLZ"}
    try:
        kids = list(d_path.parent.iterdir())
    except OSError:
        return out
    for f in sorted(kids):
        if not f.is_file() or f.suffix.lower() not in IMG_EXT:
            continue
        if f.name == d_path.name:
            continue
        m = MOD_RE.search(f.name)
        if not m:
            continue
        letter = m.group(1).upper()
        if letter in out and not out[letter]:
            out[letter].append(f)
    return out


def next_name(d_path: Path, letter: str) -> Path:
    stem = d_path.stem
    m = PREFIX_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse prefix from {d_path.name}")
    prefix = m.group(1)
    ext = d_path.suffix
    used: set[int] = set()
    for f in d_path.parent.iterdir():
        if not f.is_file():
            continue
        mm = MOD_RE.search(f.name)
        if mm and mm.group(1).upper() == letter.upper():
            num = mm.group(2)
            used.add(int(num) if num else 1)
    n = 1
    while n in used:
        n += 1
    return d_path.parent / f"{prefix}_{letter.upper()}{n}{ext}"


def apply_rename(d_path: Path, letter: str) -> Path:
    dest = next_name(d_path, letter)
    if dest.exists():
        raise FileExistsError(str(dest))
    d_path.rename(dest)
    return dest


def preview_bytes(src: Path, max_side: int, quality: int = 72) -> bytes:
    """Return JPEG preview bytes; cached on disk."""
    ensure_dirs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        mtime = int(src.stat().st_mtime)
        size = src.stat().st_size
    except OSError:
        abort(404)
    key = f"{mtime}_{size}_{max_side}_{src.name}"
    # short hash-ish folder to avoid huge names
    safe = re.sub(r"[^\w.\-]+", "_", key)[:180]
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
  <title>D-image reviewer</title>
  <style>
    :root { --bg:#0f1419; --card:#1a2332; --fg:#e7ecf3; --muted:#9aa7b8; --acc:#3d8bfd; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, Segoe UI, sans-serif; background:var(--bg); color:var(--fg); }
    header { padding:12px 16px; border-bottom:1px solid #2a3545; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    header .meta { color:var(--muted); font-size:13px; }
    main { display:grid; grid-template-columns: 1.5fr 1fr; gap:12px; padding:12px; }
    @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }
    .card { background:var(--card); border-radius:10px; padding:10px; }
    .main-img { width:100%; max-height:70vh; object-fit:contain; background:#000; border-radius:8px; }
    .thumbs { display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; }
    .thumb-wrap { background:#0b1016; border-radius:8px; padding:6px; }
    .thumb-wrap h4 { margin:0 0 4px; font-size:12px; color:var(--muted); }
    .thumb-wrap img { width:100%; height:110px; object-fit:contain; background:#000; border-radius:4px; }
    .empty { color:var(--muted); font-size:12px; padding:16px 0; text-align:center; }
    .path { font-family: ui-monospace, Consolas, monospace; font-size:11px; color:var(--muted); word-break:break-all; margin-bottom:6px; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    button, .btn {
      border:0; border-radius:8px; padding:11px 14px; font-size:15px; font-weight:700; cursor:pointer; color:#061018;
    }
    .A { background:#ff6b6b; } .G { background:#51cf66; } .L { background:#ffd43b; } .Z { background:#74c0fc; }
    .skip { background:#868e96; color:#fff; } .keep { background:#495057; color:#fff; }
    .del { background:#c92a2a; color:#fff; }
    .nav { background:var(--acc); color:#fff; text-decoration:none; }
    .hint { color:var(--muted); font-size:12px; margin-top:8px; line-height:1.4; }
    .done { padding:40px; text-align:center; }
    kbd { background:#2a3545; padding:2px 6px; border-radius:4px; font-size:11px; }
  </style>
</head>
<body>
  <header>
    <strong>Reclassify _D images</strong>
    <span class="meta">{{ index + 1 }} / {{ total }}</span>
    <span class="meta">case <b>{{ case_id }}</b></span>
    <span class="meta">done: {{ done_count }}</span>
    <a class="btn nav" href="{{ url_for('index', i=index-1) }}" {% if index<=0 %}style="pointer-events:none;opacity:.4"{% endif %}>Prev</a>
    <a class="btn nav" href="{{ url_for('index', i=index+1) }}" {% if index>=total-1 %}style="pointer-events:none;opacity:.4"{% endif %}>Next</a>
  </header>

  {% if total == 0 %}
  <div class="done"><h2>No _D images left</h2></div>
  {% else %}
  <main>
    <section class="card">
      <div class="path">{{ rel }}</div>
      <img class="main-img" src="{{ url_for('image', i=index, kind='d') }}" alt="D" decoding="async"/>
      {% if index + 1 < total %}
      <link rel="prefetch" href="{{ url_for('image', i=index+1, kind='d') }}"/>
      {% endif %}
      <div class="actions">
        <form method="post" action="{{ url_for('decide') }}" style="display:contents">
          <input type="hidden" name="index" value="{{ index }}"/>
          <button class="A" name="action" value="A">A</button>
          <button class="G" name="action" value="G">G</button>
          <button class="L" name="action" value="L">L</button>
          <button class="Z" name="action" value="Z">Z</button>
          <button class="keep" name="action" value="KEEP">Keep D</button>
          <button class="skip" name="action" value="SKIP">Skip</button>
          <button class="del" name="action" value="DELETE"
                  onclick="return confirm('Delete this _D image permanently?');">Delete</button>
        </form>
      </div>
      <p class="hint">
        Rename to next free slot. Keys: <kbd>A</kbd> <kbd>G</kbd> <kbd>L</kbd> <kbd>Z</kbd> ·
        <kbd>K</kbd> keep · <kbd>S</kbd> skip · <kbd>X</kbd> delete · arrows
      </p>
      {% if preview_name %}<p class="hint">Example if A: <code>{{ preview_name }}</code></p>{% endif %}
    </section>

    <section class="card">
      <h3 style="margin:2px 0 8px;font-size:14px;">Compare A/G/L/Z in same folder</h3>
      <div class="thumbs">
        {% for letter in ['A','G','L','Z'] %}
        <div class="thumb-wrap">
          <h4>{{ letter }}{% if refs[letter] %} — {{ refs[letter][0] }}{% endif %}</h4>
          {% if refs[letter] %}
            <img src="{{ url_for('image', i=index, kind='ref', letter=letter, j=0) }}" alt="{{ letter }}" decoding="async"/>
          {% else %}
            <div class="empty">none</div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </section>
  </main>
  {% endif %}

<script>
document.addEventListener('keydown', (e) => {
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  const map = {a:'A', g:'G', l:'L', z:'Z', k:'KEEP', s:'SKIP', x:'DELETE'};
  const key = e.key.toLowerCase();
  if (map[key]) {
    e.preventDefault();
    const form = document.querySelector('form');
    if (!form) return;
    const btn = form.querySelector(`button[value="${map[key]}"]`);
    if (btn) btn.click();
  } else if (e.key === 'ArrowRight') {
    location.href = "{{ url_for('index', i=index+1) }}";
  } else if (e.key === 'ArrowLeft') {
    location.href = "{{ url_for('index', i=index-1) }}";
  }
});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    files = get_files()
    state = load_state()
    total = len(files)
    if total == 0:
        return render_template_string(
            PAGE, index=0, total=0, case_id="-", rel="", refs={k: [] for k in "AGLZ"},
            done_count=len(state.get("done", [])), preview_name=""
        )
    i = int(request.args.get("i", state.get("index", 0)))
    i = max(0, min(total - 1, i))
    state["index"] = i
    save_state(state)
    path = Path(files[i])
    if not path.exists():
        # stale entry — rescan
        refresh_index()
        return redirect(url_for("index", i=i))
    refs_paths = sibling_mods(path)
    refs = {k: [p.name for p in v] for k, v in refs_paths.items()}
    try:
        preview = next_name(path, "A").name
    except Exception:
        preview = ""
    return render_template_string(
        PAGE,
        index=i,
        total=total,
        case_id=case_id_of(path),
        rel=str(path.relative_to(DATA)),
        refs=refs,
        done_count=len(state.get("done", [])),
        preview_name=preview,
    )


@app.get("/img")
def image():
    files = get_files()
    i = int(request.args.get("i", 0))
    if i < 0 or i >= len(files):
        abort(404)
    d_path = Path(files[i])
    kind = request.args.get("kind", "d")
    if kind == "d":
        if not d_path.exists():
            abort(404)
        data = preview_bytes(d_path, max_side=1280, quality=75)
        return send_file(io.BytesIO(data), mimetype="image/jpeg", max_age=3600)

    letter = request.args.get("letter", "A").upper()
    j = int(request.args.get("j", 0))
    refs = sibling_mods(d_path).get(letter, [])
    if j < 0 or j >= len(refs):
        abort(404)
    data = preview_bytes(refs[j], max_side=480, quality=70)
    return send_file(io.BytesIO(data), mimetype="image/jpeg", max_age=3600)


@app.post("/decide")
def decide():
    files = get_files()
    i = int(request.form.get("index", 0))
    action = (request.form.get("action") or "SKIP").upper()
    state = load_state()
    if i < 0 or i >= len(files):
        return redirect(url_for("index", i=0))
    path = Path(files[i])
    if not path.exists():
        refresh_index()
        return redirect(url_for("index", i=i))
    rel = str(path.relative_to(DATA))

    if action in {"A", "G", "L", "Z"}:
        try:
            dest = apply_rename(path, action)
            log_event({"action": action, "from": rel, "to": str(dest.relative_to(DATA))})
            state.setdefault("done", []).append({"from": rel, "to": str(dest.relative_to(DATA)), "as": action})
            with _lock:
                if i < len(_files) and _files[i] == str(path):
                    _files.pop(i)
        except Exception as e:
            log_event({"action": "ERROR", "from": rel, "error": str(e)})
            save_state(state)
            return f"<pre>Rename failed: {e}\n<a href='{url_for('index', i=i)}'>Back</a></pre>", 500
        save_state(state)
        return redirect(url_for("index", i=min(i, max(0, len(get_files()) - 1))))

    if action == "DELETE":
        try:
            path.unlink()
            log_event({"action": "DELETE", "from": rel})
            state.setdefault("done", []).append({"from": rel, "as": "DELETE"})
            with _lock:
                if i < len(_files) and _files[i] == str(path):
                    _files.pop(i)
        except Exception as e:
            log_event({"action": "ERROR", "from": rel, "error": str(e)})
            save_state(state)
            return f"<pre>Delete failed: {e}\n<a href='{url_for('index', i=i)}'>Back</a></pre>", 500
        save_state(state)
        return redirect(url_for("index", i=min(i, max(0, len(get_files()) - 1))))

    if action == "KEEP":
        log_event({"action": "KEEP", "from": rel})
        state.setdefault("done", []).append({"from": rel, "as": "KEEP"})
        save_state(state)
        return redirect(url_for("index", i=i + 1))

    log_event({"action": "SKIP", "from": rel})
    save_state(state)
    return redirect(url_for("index", i=i + 1))


def main():
    ensure_dirs()
    if not DATA.is_dir():
        raise SystemExit(f"Data folder not found: {DATA}")
    print("Scanning _D images (one-time)...")
    refresh_index()
    print(f"Orginal Data: {DATA}")
    print(f"_D images found: {len(get_files())}")
    print(f"Preview cache: {CACHE_DIR}")
    print("Open: http://127.0.0.1:8765")
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Flag likely blurry images under data/curated/My_Dataset.

Two scores are computed on a central crop (the dark circular border is left
out, because it is almost flat and would fake a blur score):

* laplacian — variance of the Laplacian after a fixed resize. Lower means
  fewer sharp edges. Smooth in-focus tissue also scores low, so this alone
  is not the flag.
* edge_blur — Crete perceptual blur, measured only on the strong edges.
  0 is sharp, 1 is already blurry (blurring the photo again barely changes
  those edges). This is the flag.

An image is listed for review when either of these is true:

* edge_blur >= 0.50 and laplacian < 450 — soft overview frame (acetic,
  green, or Lugol) whose edges are already wide, and the photo is not a
  sharp high-detail Lugol.
* laplacian < 90 and edge_blur >= 0.18 — defocused zoom / very soft frame.
  The edge_blur floor keeps smooth but in-focus zooms off the list.

Check the files before deleting or excluding them. A few hazy-but-usable
frames can still appear.

Usage (conda env `colpo`):

    python ml/scripts/detect_blur.py
    python ml/scripts/detect_blur.py --threshold 0.40
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT / "data" / "curated" / "My_Dataset"
DEFAULT_OUT = PROJECT / "ml" / "runs" / "blur_audit"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Tuned on full-size checks in this cohort. See the module docstring.
EDGE_BLUR_OVERVIEW = 0.50
LAPLACIAN_OVERVIEW_MAX = 450.0
LAPLACIAN_SOFT = 90.0
EDGE_BLUR_SOFT_MIN = 0.18


def is_likely_blur(edge_blur: float, laplacian: float) -> bool:
    if edge_blur != edge_blur:
        return False
    soft_overview = edge_blur >= EDGE_BLUR_OVERVIEW and laplacian < LAPLACIAN_OVERVIEW_MAX
    soft_detail = laplacian < LAPLACIAN_SOFT and edge_blur >= EDGE_BLUR_SOFT_MIN
    return soft_overview or soft_detail


def read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception:
        return None


def central_patch(gray: np.ndarray, crop: float = 0.12) -> np.ndarray:
    h, w = gray.shape[:2]
    y0, y1 = int(h * crop), int(h * (1.0 - crop))
    x0, x1 = int(w * crop), int(w * (1.0 - crop))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return gray
    return gray[y0:y1, x0:x1]


def resize_short(patch: np.ndarray, side: int) -> np.ndarray:
    ph, pw = patch.shape[:2]
    short = min(ph, pw)
    if short < 8:
        return patch
    scale = side / float(short)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(patch, None, fx=scale, fy=scale, interpolation=interp)


def focus_score(gray: np.ndarray, side: int = 384) -> float:
    """Laplacian variance on a fixed-size central crop. Lower = fewer edges."""
    resized = resize_short(central_patch(gray), side)
    if min(resized.shape[:2]) < 8:
        return 0.0
    lap = cv2.Laplacian(resized, cv2.CV_64F, ksize=3)
    return float(lap.var())


def edge_blur_score(gray: np.ndarray, side: int = 512, kernel: int = 9) -> float:
    """Crete blur on strong edges only. 0 = sharp, 1 = edges already soft."""
    patch = central_patch(gray).astype(np.float32)
    image = resize_short(patch, side)
    if min(image.shape[:2]) < 16:
        return float("nan")
    blurred = cv2.blur(image, (kernel, kernel))
    scores: list[float] = []
    for axis in (0, 1):
        original = np.abs(np.diff(image, axis=axis))
        after = np.abs(np.diff(blurred, axis=axis))
        lost = np.maximum(0.0, original - after)
        cutoff = max(float(np.percentile(original, 80)), 1.0)
        strong = original >= cutoff
        if int(strong.sum()) < 40:
            continue
        kept = float(lost[strong].sum() / (original[strong].sum() + 1e-6))
        scores.append(1.0 - kept)
    if not scores:
        return float("nan")
    return float(max(scores))


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def save_contact_sheet(
    rows: list[dict],
    out_path: Path,
    cols: int = 6,
    thumb: int = 180,
) -> None:
    if not rows:
        return
    thumbs: list[np.ndarray] = []
    for row in rows:
        gray = read_gray(Path(row["path"]))
        if gray is None:
            tile = np.zeros((thumb, thumb), np.uint8)
        else:
            h, w = gray.shape[:2]
            scale = thumb / float(max(h, w))
            tile = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((thumb, thumb), np.uint8)
            y = (thumb - tile.shape[0]) // 2
            x = (thumb - tile.shape[1]) // 2
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
            tile = canvas
        label = f"{row['patient']}  {row['edge_blur']:.2f}"
        color = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(color, label, (4, thumb - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(color)
    rows_n = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows_n * thumb, cols * thumb, 3), np.uint8)
    for i, tile in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * thumb : (r + 1) * thumb, c * thumb : (c + 1) * thumb] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "edge_blur",
        "laplacian",
        "patient",
        "file",
        "rel",
        "path",
        "width",
        "height",
        "flagged",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect blurry images in My_Dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    root = args.root.resolve()
    paths = sorted(iter_images(root))
    print(f"images: {len(paths)} under {root}")

    scored: list[dict] = []
    failed: list[str] = []
    for i, path in enumerate(paths, start=1):
        gray = read_gray(path)
        if gray is None:
            failed.append(str(path))
            continue
        lap = focus_score(gray)
        blur = edge_blur_score(gray)
        try:
            rel = path.relative_to(root)
            patient = rel.parts[1] if len(rel.parts) > 1 and rel.parts[0] == "patients" else rel.parts[0]
        except ValueError:
            rel = Path(path.name)
            patient = path.parent.name
        scored.append(
            {
                "edge_blur": blur,
                "laplacian": lap,
                "patient": patient,
                "file": path.name,
                "rel": str(rel),
                "path": str(path),
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
            }
        )
        if i % 400 == 0 or i == len(paths):
            print(f"scored {i}/{len(paths)}")

    scored.sort(key=lambda row: row["edge_blur"] if row["edge_blur"] == row["edge_blur"] else -1.0, reverse=True)
    for rank, row in enumerate(scored, start=1):
        blur = row["edge_blur"]
        lap = row["laplacian"]
        row["rank"] = rank
        row["edge_blur"] = round(float(blur), 4) if blur == blur else ""
        row["laplacian"] = round(float(lap), 2)
        row["flagged"] = int(is_likely_blur(float(blur), float(lap)))

    flagged = [row for row in scored if row["flagged"]]
    out = args.out.resolve()
    write_csv(out / "blur_scores.csv", scored)
    write_csv(out / "blurry_review.csv", flagged)
    save_contact_sheet(flagged[:48], out / "contact_blurriest48.jpg")

    blur_vals = np.array(
        [row["edge_blur"] for row in scored if row["edge_blur"] != ""],
        dtype=np.float64,
    )
    pcts = [50, 75, 90, 95, 98, 99]
    lines = [
        "# Blur review — My_Dataset",
        "",
        "edge_blur is the Crete perceptual blur score on the strong edges of the",
        "central crop (0 = sharp, 1 = already blurry). laplacian is the variance",
        "of the Laplacian; low values also happen on smooth in-focus tissue, so",
        "the review list uses edge_blur only.",
        "",
        f"- Images scored: {len(scored)}",
        f"- Unreadable: {len(failed)}",
        "- Flag rule: edge_blur >= 0.50 and laplacian < 450,",
        "  or laplacian < 90 and edge_blur >= 0.18",
        f"- Flagged for visual check: {len(flagged)}",
        "",
        "## edge_blur percentiles (higher = blurrier)",
        "",
    ]
    for p in pcts:
        lines.append(f"- p{p}: {float(np.percentile(blur_vals, p)):.3f}")
    lines.extend(["", "## Flagged images (blurriest first)", ""])
    for row in flagged:
        lines.append(
            f"- {float(row['edge_blur']):.3f}  lap {float(row['laplacian']):.0f}  `{row['rel']}`"
        )
    if failed:
        lines.extend(["", "## Unreadable", ""])
        lines.extend(f"- `{p}`" for p in failed)
    (out / "BLUR_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"flagged: {len(flagged)}")
    print(f"wrote {out / 'blur_scores.csv'}")
    print(f"wrote {out / 'blurry_review.csv'}")
    print(f"wrote {out / 'BLUR_REVIEW.md'}")


if __name__ == "__main__":
    main()

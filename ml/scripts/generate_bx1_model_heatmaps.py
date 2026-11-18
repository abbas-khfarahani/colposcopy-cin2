"""Generate probability tables and Grad-CAM galleries for BX1-gold models.

Modes:
  efficientnet_b3: frozen holdout for acetic_efficientnet_b3_stable.
  b1: fold-specific OOF inference for protocol B1 ResNet50 GAP.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml.boost import load_ckpt
from ml.data import build_transforms
from ml.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from ml.protocol.datasets import open_rgb
from ml.protocol.encoders import build_protocol_model
from ml.protocol.transforms import build_protocol_transforms


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self.fh = layer.register_forward_hook(self._forward)
        self.bh = layer.register_full_backward_hook(self._backward)

    def _forward(self, _m, _i, output) -> None:
        self.activations = output.detach()

    def _backward(self, _m, _gi, output) -> None:
        self.gradients = output[0].detach()

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        logits.sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        maximum = cam.max().detach()
        if float(maximum) > 1e-8:
            cam = cam / maximum
        return float(torch.sigmoid(logits[0]).detach().cpu()), cam.detach().cpu().numpy()

    def close(self) -> None:
        self.fh.remove()
        self.bh.remove()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return ((x * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps
    return (colormaps["jet"](np.clip(cam, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    return (0.55 * rgb + 0.45 * heatmap_rgb(cam)).astype(np.uint8)


def save_triplet(rgb: np.ndarray, cam: np.ndarray, original: Path, heat: Path, over: Path) -> None:
    Image.fromarray(rgb).save(original)
    Image.fromarray(heatmap_rgb(cam)).save(heat)
    Image.fromarray(overlay(rgb, cam)).save(over)


def save_images(rgb: np.ndarray, cam: np.ndarray, title: str, panel: Path, over: Path) -> None:
    """Legacy three-in-one panel used by the B1 out-of-fold gallery."""
    ov = overlay(rgb, cam)
    Image.fromarray(ov).save(over)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(rgb); axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Grad-CAM")
    axes[2].imshow(ov); axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(panel, dpi=120, bbox_inches="tight")
    plt.close(fig)


def setup_out(name: str) -> tuple[Path, Path, Path, Path]:
    out = PROJECT / "ml/runs" / name
    originals, heatmaps, overlays = out / "originals", out / "heatmaps", out / "overlays"
    for folder in (originals, heatmaps, overlays, out / "panels"):
        folder.mkdir(parents=True, exist_ok=True)
    return out, originals, heatmaps, overlays


HOLDOUT_GALLERIES = {
    "efficientnet_b3": {
        "dir": "heatmaps_efficientnet_b3_stable_test",
        "run": "acetic_efficientnet_b3_stable",
        "title": "EfficientNet-B3 — holdout test",
        "threshold": 0.06743042916059494,
        "reported_auroc": 0.832,
    },
    "resnet50_384": {
        "dir": "heatmaps_resnet50_stable_384_test",
        "run": "acetic_resnet50_stable_384",
        "title": "ResNet50 at 384 px — holdout test",
        "threshold": 0.07423318922519684,
        "reported_auroc": 0.831,
    },
}


def _label(flag: int) -> str:
    return "CIN2+" if int(flag) else "CIN1-"


def _holdout_card(row: dict, person: dict, threshold: float) -> str:
    probability = float(row["probability_cin2plus"])
    person_probability = float(person["probability_cin2plus_max"])
    true_label = _label(int(row["label"]))
    predicted = _label(probability >= threshold)
    person_predicted = _label(person_probability >= threshold)
    outcome = "match" if predicted == true_label else "miss"
    return (
        '<article class="{outcome}">'
        '<div class="views">'
        '<figure><img src="{original}" alt="Original" loading="lazy"><figcaption>Original</figcaption></figure>'
        '<figure><img src="{heatmap}" alt="Heatmap" loading="lazy"><figcaption>Heatmap</figcaption></figure>'
        '<figure><img src="{overlay}" alt="Overlay" loading="lazy"><figcaption>Overlay</figcaption></figure>'
        '<figure><img src="{segmentation}" alt="Segmentation" loading="lazy"><figcaption>Segmentation mask</figcaption></figure>'
        "</div>"
        "<p><b>Person</b> {pid} &nbsp; <b>Image</b> {file}</p>"
        "<p><b>True label:</b> {true_label}</p>"
        "<p><b>Predicted label:</b> {predicted} &nbsp; <b>Probability:</b> {probability:.4f}</p>"
        "<p><b>Person prediction</b> (max image): {person_predicted} &nbsp; <b>Person probability:</b> {person_probability:.4f}</p>"
        "</article>"
    ).format(
        outcome=outcome,
        original=html.escape(row["original"]),
        heatmap=html.escape(row["heatmap"]),
        overlay=html.escape(row["overlay"]),
        segmentation=html.escape(row["original"].replace("originals/", "segmentation/", 1)),
        pid=html.escape(str(row["person_id"])),
        file=html.escape(row["file"]),
        true_label=html.escape(true_label),
        predicted=html.escape(predicted),
        probability=probability,
        person_predicted=html.escape(person_predicted),
        person_probability=person_probability,
    )


def rebuild_html(mode: str) -> None:
    if mode == "b1":
        _rebuild_b1_html()
        return
    spec = HOLDOUT_GALLERIES[mode]
    out = PROJECT / "ml/runs" / spec["dir"]
    image_rows = read_csv(out / "per_image_probabilities.csv")
    people = {str(r["person_id"]): r for r in read_csv(out / "per_person_probabilities.csv")}
    threshold = float(spec["threshold"])
    gallery = "\n".join(_holdout_card(r, people[str(r["person_id"])], threshold) for r in image_rows)
    ranked = sorted(image_rows, key=lambda r: float(r["probability_cin2plus"]), reverse=True)
    by_probability = "\n".join(_holdout_card(r, people[str(r["person_id"])], threshold) for r in ranked)
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
body{{font-family:Arial,sans-serif;margin:0;background:#f3f4f6;color:#111}}
header,section{{padding:20px 24px}}
header{{background:#111827;color:#fff;position:sticky;top:0;z-index:2}}
header a{{color:#fde68a;margin-right:16px}}
h1{{margin:0 0 8px;font-size:22px}}
h2{{margin:8px 0 4px}}
.note{{margin:0;color:#d1d5db}}
article{{background:#fff;padding:14px;margin:14px 0;border-radius:10px;border-left:6px solid #9ca3af}}
article.match{{border-left-color:#15803d}}
article.miss{{border-left-color:#b91c1c}}
.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
figure{{margin:0;min-width:0}}
img{{width:100%;max-width:100%;height:auto;background:#111;border-radius:6px;display:block}}
figcaption{{font-size:13px;margin-top:4px;color:#374151}}
p{{margin:6px 0}}
</style></head><body>
<header>
<h1>{title}</h1>
<p class="note">{n_images} test images, {n_persons} people. Predicted label uses this model's saved threshold {threshold:.6f}. Person AUC on this holdout: {auroc:.3f}. A green bar means the image prediction matches the true label. A red bar means it does not.</p>
<p class="note">Segmentation is the automatic color mask, not a hand drawing. Lilac = thin acetowhite, magenta = dense acetowhite, red = red tissue, yellow = glare.</p>
<p><a href="#cases">1. Test cases</a><a href="#ranked">2. Highest probability to lowest</a></p>
</header>
<section id="cases"><h2>1. Test cases</h2>
<p>Original image, Grad-CAM heatmap, and overlay, with the true label, the predicted label, and the CIN2+ probability.</p>
{gallery}
</section>
<section id="ranked"><h2>2. Sorted from highest probability to lowest</h2>
<p>The same test images, ordered by this image's CIN2+ probability.</p>
{by_probability}
</section>
</body></html>
""".format(
        title=html.escape(spec["title"]),
        n_images=len(image_rows),
        n_persons=len(people),
        threshold=threshold,
        auroc=float(spec["reported_auroc"]),
        gallery=gallery,
        by_probability=by_probability,
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def _rebuild_b1_html() -> None:
    out = PROJECT / "ml/runs/protocol/heatmaps_B1_seed42_oof"
    title_name = "B1 ResNet50 GAP — fold-specific OOF"
    image_threshold = 0.5
    image_rows = read_csv(out / "per_image_probabilities.csv")
    people = {str(r["person_id"]): r for r in read_csv(out / "per_person_probabilities.csv")}
    cards = []
    for r in image_rows:
        pid = str(r["person_id"])
        person = people[pid]
        ip = float(r["probability_cin2plus"])
        image_pred = _label(ip >= image_threshold)
        person_pred = _label(int(person["predicted_class_op70"]))
        person_probability = float(person["calibrated_probability"])
        true_label = _label(int(r["label"]))
        cards.append(
            '<article><a href="{panel}"><img src="{panel}" loading="lazy"></a>'
            "<p><b>{pid}</b> / {file} / {fold}</p>"
            "<p><b>True label:</b> {true_label} &nbsp; "
            "<b>Image prediction:</b> {image_pred} (p={ip:.4f}) &nbsp; "
            "<b>Person prediction:</b> {person_pred} (calibrated p={pp:.4f}, OP70)</p></article>".format(
                panel=html.escape(r["panel"]),
                pid=html.escape(pid),
                file=html.escape(r["file"]),
                fold=html.escape(str(r["outer_fold"])),
                true_label=true_label,
                image_pred=image_pred,
                ip=ip,
                person_pred=person_pred,
                pp=person_probability,
            )
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>"
        + html.escape(title_name)
        + "</title><style>body{font-family:Arial;margin:24px;background:#f5f5f5}"
        "article{background:#fff;padding:12px;margin:12px 0;border-radius:8px}img{max-width:1100px;width:100%}</style>"
        f"<h1>{html.escape(title_name)}</h1>"
        "<p>Image display threshold 0.5. Official person decision uses calibrated OP70.</p>"
        + "\n".join(cards)
    )
    (out / "index.html").write_text(page, encoding="utf-8")


def cam_target(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    if backbone.startswith("efficientnet"):
        return model.backbone.features[-1]
    if backbone.startswith("resnet"):
        return model.backbone.layer4
    raise ValueError(f"No Grad-CAM layer for backbone {backbone}")


def holdout_rows(run_name: str, device: torch.device):
    run = PROJECT / "ml/runs" / run_name
    model, cfg = load_ckpt(run / "best.pt", device)
    backbone = str(cfg.get("backbone", "resnet50"))
    cam = GradCAM(model, cam_target(model, backbone))
    tf = build_transforms("val", int(cfg["image_size"]), False)
    source = json.loads((run / "test_predictions.json").read_text(encoding="utf-8"))
    rows = []
    for r in source:
        raw = str(r["path"]).replace("\\", "/")
        rel = "patients/" + raw.split("/patients/", 1)[1]
        rows.append({
            "person_id": str(r["person_id"]), "folder_id": str(r["folder_id"]),
            "file": str(r["file"]), "image_path_rel": rel, "label": int(r["label"]),
            "stored_probability": float(r["score"]), "outer_fold": "",
        })
    return model, cam, tf, rows, PROJECT / "data/curated/My_Dataset", cfg


def load_b1_fold(fold: int, device: torch.device):
    path = PROJECT / f"ml/runs/protocol/experiments/B1/seed42/fold_{fold}/best.pt"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_protocol_model(backbone="resnet50", model_type="image", pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, GradCAM(model, model.net.backbone.layer4)


def make_gallery(mode: str, device: torch.device) -> None:
    if mode in HOLDOUT_GALLERIES:
        spec = HOLDOUT_GALLERIES[mode]
        out, originals, heatmaps, overlays = setup_out(spec["dir"])
        model, cam, tf, rows, root, cfg = holdout_rows(spec["run"], device)
        models, cams = {0: model}, {0: cam}
        title_name = spec["title"]
        person_reduce = "max"
        expected_metrics = {"person_auroc": spec["reported_auroc"], "threshold": spec["threshold"]}
    else:
        out, originals, heatmaps, overlays = setup_out("protocol/heatmaps_B1_seed42_oof")
        rows = read_csv(PROJECT / "ml/runs/protocol/experiments/B1/seed42/oof_image_predictions.csv")
        tf = build_protocol_transforms("eval", 224)
        root = PROJECT / "data/curated/My_Dataset"
        models, cams, cfg = {}, {}, {}
        title_name = "B1 ResNet50 GAP — fold-specific OOF"
        person_reduce = "max"
        expected_metrics = {"person_auroc": 0.7096184813576116, "person_auprc": 0.36302847339401606}

    output_rows = []
    person_values = defaultdict(list)
    person_labels = {}
    try:
        for i, row in enumerate(rows, 1):
            fold = int(row.get("outer_fold") or 0)
            if mode == "b1" and fold not in models:
                models[fold], cams[fold] = load_b1_fold(fold, device)
            path = root / row["image_path_rel"]
            pil = open_rgb(path)
            x0 = tf(pil)
            recomputed_probability, heatmap = cams[fold](x0.unsqueeze(0).to(device))
            rgb = denorm(x0)
            pid = str(row["person_id"])
            label = int(row.get("person_label", row["label"]))
            stored = float(row.get("probability", row.get("stored_probability", recomputed_probability)))
            probability = stored
            safe = f"{i:04d}_{pid.replace('.', '_')}_{Path(row['file']).stem}"
            original_rel = f"originals/{safe}.png"
            heatmap_rel = f"heatmaps/{safe}.png"
            overlay_rel = f"overlays/{safe}.png"
            panel_rel = f"panels/{safe}.png"
            if mode == "b1":
                title = f"person={pid} | y={label} | p(CIN2+)={probability:.4f} | fold={fold}"
                save_images(rgb, heatmap, title, out / panel_rel, out / overlay_rel)
            else:
                save_triplet(rgb, heatmap, out / original_rel, out / heatmap_rel, out / overlay_rel)
            output_rows.append({
                "person_id": pid, "folder_id": row["folder_id"], "file": row["file"],
                "image_path_rel": row["image_path_rel"], "label": label,
                "probability_cin2plus": probability,
                "probability_recomputed_during_cam": recomputed_probability,
                "absolute_probability_difference": abs(recomputed_probability - stored),
                "outer_fold": fold if mode == "b1" else "holdout",
                "original": original_rel, "heatmap": heatmap_rel,
                "overlay": overlay_rel, "panel": panel_rel,
            })
            person_values[pid].append(probability)
            person_labels[pid] = max(person_labels.get(pid, 0), label)
            print(f"[{i}/{len(rows)}] {pid} {row['file']} p={probability:.4f}", flush=True)
    finally:
        for cam in cams.values(): cam.close()

    person_rows = []
    for pid in sorted(person_values):
        vals = person_values[pid]
        person_rows.append({
            "person_id": pid, "label": person_labels[pid], "n_images": len(vals),
            "probability_cin2plus_max": max(vals),
        })
    if mode == "b1":
        official = {
            str(r["person_id"]): r
            for r in read_csv(PROJECT / "ml/runs/protocol/experiments/B1/seed42/oof_predictions.csv")
        }
        for row in person_rows:
            saved = official[row["person_id"]]
            row["calibrated_probability"] = float(saved["calibrated_probability"])
            row["predicted_class_op70"] = int(saved["predicted_class_op70"])
            row["predicted_class_op80"] = int(saved["predicted_class_op80"])
    y = [r["label"] for r in person_rows]
    p = [r["probability_cin2plus_max"] for r in person_rows]
    metrics = {"person_auroc_recomputed_raw": roc_auc_score(y, p), "person_auprc_recomputed_raw": average_precision_score(y, p)}
    if mode == "b1":
        calibrated = [r["calibrated_probability"] for r in person_rows]
        metrics["person_auroc_official_calibrated"] = roc_auc_score(y, calibrated)
        metrics["person_auprc_official_calibrated"] = average_precision_score(y, calibrated)
    write_csv(out / "per_image_probabilities.csv", output_rows)
    write_csv(out / "per_person_probabilities.csv", person_rows)
    summary = {
        "model": title_name, "evaluation": "holdout" if mode in HOLDOUT_GALLERIES else "out_of_fold",
        "dataset": "My_Dataset BX1-gold", "n_persons": len(person_rows), "n_images": len(output_rows),
        "aggregation": person_reduce, "metrics": metrics, "reported_metrics": expected_metrics,
        "max_probability_reproduction_error": max(r["absolute_probability_difference"] for r in output_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rebuild_html(mode)
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["efficientnet_b3", "resnet50_384", "b1"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--html-only", action="store_true", help="Rebuild HTML from existing CSVs without inference")
    args = ap.parse_args()
    if args.html_only:
        rebuild_html(args.model)
        return
    make_gallery(args.model, torch.device(args.device))


if __name__ == "__main__":
    main()

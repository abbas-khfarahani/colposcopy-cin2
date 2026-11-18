"""Exploratory Grad-CAM and probabilities for Stage-1 F50 on My_Dataset test cases.

This is intentionally separate from the locked initial-protocol final-test path.
It averages the three seed-17 F50 (lr=3e-5) fold checkpoints selected during
development. Case probability follows Stage 1's ALL_AA_PROB_MEAN rule: mean of
per-image sigmoid probabilities.
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
from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ml.initial_protocol_v1.shared.decode import decode_rgb
from ml.initial_protocol_v1.shared.images import eval_view
from ml.initial_protocol_v1.stage1.models import Stage1Classifier


RUNS = [
    "F50_r3e-05_bce_A1_wd0.0001_fold0_seed17__ccb2443b3dbf",
    "F50_r3e-05_bce_A1_wd0.0001_fold1_seed17__025e93e5f5fb",
    "F50_r3e-05_bce_A1_wd0.0001_fold2_seed17__d9f60c328f61",
]


class GradCAM:
    def __init__(self, model: Stage1Classifier) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer = model.encoder.layer4
        self._forward = layer.register_forward_hook(self._save_activations)
        self._backward = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)[0]
        probability = float(torch.sigmoid(logit).detach().cpu())
        logit.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        if float(cam.max()) > 0:
            cam /= cam.max()
        return probability, cam.detach().cpu().numpy()

    def close(self) -> None:
        self._forward.remove()
        self._backward.remove()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_model(checkpoint: Path, device: torch.device) -> Stage1Classifier:
    encoder = resnet50(weights=None)
    encoder.fc = torch.nn.Identity()
    model = Stage1Classifier(encoder, 2048)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model


def save_panel(rgb: np.ndarray, cam: np.ndarray, title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("F50 ensemble Grad-CAM")
    axes[2].imshow(rgb)
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_overlay(rgb: np.ndarray, cam: np.ndarray, out: Path) -> None:
    """Save a clean heatmap-on-image PNG like the existing *_panel_overlay files."""
    h, w = rgb.shape[:2]
    cam_native = F.interpolate(
        torch.from_numpy(cam).float()[None, None],
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    heat_rgb = colormaps["jet"](np.clip(cam_native, 0, 1))[..., :3]
    blended = np.clip(0.55 * rgb + 0.45 * heat_rgb, 0, 1)
    Image.fromarray(np.uint8(np.round(blended * 255))).save(out)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="ml/runs/initial_protocol_v1/stage1/f50_test_mydataset_heatmaps")
    args = ap.parse_args()

    device = torch.device(args.device)
    out = PROJECT / args.out_dir
    panels = out / "panels"
    overlays = out / "overlays"
    panels.mkdir(parents=True, exist_ok=True)
    overlays.mkdir(parents=True, exist_ok=True)

    split_rows = read_csv(PROJECT / "ml/runs/initial_protocol_v1/splits/master_split.csv")
    test_keys = {
        r["case_key"] for r in split_rows
        if r["split"] == "test" and r["dataset"] == "My_Dataset"
    }
    cases = {r["case_key"]: r for r in read_csv(PROJECT / "ml/runs/initial_protocol_v1/data/case_table.csv")}
    images = [
        r for r in read_csv(PROJECT / "ml/runs/initial_protocol_v1/data/case_images_retained.csv")
        if r["case_key"] in test_keys and r["image_code"].upper().startswith("AA")
    ]
    images.sort(key=lambda r: (r["case_key"], r["image_code"], r["relative_path"]))

    ckpts = [
        PROJECT / "ml/runs/initial_protocol_v1/stage1/experiments" / run / "checkpoints/best.pt"
        for run in RUNS
    ]
    missing = [str(p) for p in ckpts if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints: " + ", ".join(missing))
    models = [load_model(p, device) for p in ckpts]
    cams = [GradCAM(m) for m in models]

    per_image: list[dict] = []
    by_case: dict[str, list[float]] = defaultdict(list)
    try:
        for idx, row in enumerate(images, 1):
            source = PROJECT / "data" / row["relative_path"]
            decoded = decode_rgb(source)
            x = eval_view(decoded.rgb, size=224).unsqueeze(0).to(device)
            probs, heatmaps = zip(*(cam(x) for cam in cams))
            probability = float(np.mean(probs))
            heatmap = np.mean(np.stack(heatmaps), axis=0)
            if float(heatmap.max()) > 0:
                heatmap /= heatmap.max()
            meta = cases[row["case_key"]]
            safe = f"{idx:03d}_{meta['case_id'].replace('.', '_')}_{row['image_code']}"
            panel_rel = f"panels/{safe}.png"
            overlay_rel = f"overlays/{safe}_panel_overlay.png"
            title = (
                f"case={meta['case_id']} | image={row['image_code']} | "
                f"label={meta['label']} ({meta['grade']}) | p(CIN2+)={probability:.4f}"
            )
            save_panel(decoded.rgb, heatmap, title, out / panel_rel)
            save_overlay(decoded.rgb, heatmap, out / overlay_rel)
            item = {
                "case_key": row["case_key"],
                "case_id": meta["case_id"],
                "patient_group_id": meta["patient_group_id"],
                "label": int(meta["label"]),
                "grade": meta["grade"],
                "image_code": row["image_code"],
                "relative_path": row["relative_path"],
                "probability_cin2plus": probability,
                "predicted_label_at_0_5": int(probability >= 0.5),
                "fold0_probability": probs[0],
                "fold1_probability": probs[1],
                "fold2_probability": probs[2],
                "panel": panel_rel,
                "overlay": overlay_rel,
            }
            per_image.append(item)
            by_case[row["case_key"]].append(probability)
            print(f"[{idx}/{len(images)}] {meta['case_id']} {row['image_code']} p={probability:.4f}", flush=True)
    finally:
        for cam in cams:
            cam.close()

    per_case = []
    for key in sorted(by_case):
        meta = cases[key]
        values = by_case[key]
        per_case.append({
            "case_key": key,
            "case_id": meta["case_id"],
            "patient_group_id": meta["patient_group_id"],
            "label": int(meta["label"]),
            "grade": meta["grade"],
            "n_aa_images": len(values),
            "probability_cin2plus_all_aa_mean": float(np.mean(values)),
            "predicted_label_at_0_5": int(float(np.mean(values)) >= 0.5),
        })

    image_fields = list(per_image[0]) if per_image else []
    case_fields = list(per_case[0]) if per_case else []
    write_csv(out / "per_image_probabilities.csv", per_image, image_fields)
    write_csv(out / "per_case_probabilities.csv", per_case, case_fields)

    summary = {
        "status": "exploratory_not_protocol_final",
        "dataset": "My_Dataset",
        "split": "test",
        "eligibility": "strict_AA",
        "model": "mean ensemble of three Stage-1 F50 lr=3e-5 seed17 fold checkpoints",
        "case_probability": "mean of ensemble per-image sigmoid probabilities (ALL_AA_PROB_MEAN)",
        "cam": "mean layer4 Grad-CAM across the same three checkpoints",
        "n_cases": len(per_case),
        "n_images": len(per_image),
        "checkpoints": [str(p.relative_to(PROJECT)) for p in ckpts],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cards = []
    case_probs = {r["case_key"]: r["probability_cin2plus_all_aa_mean"] for r in per_case}
    case_preds = {r["case_key"]: r["predicted_label_at_0_5"] for r in per_case}
    for r in per_image:
        cards.append(
            '<article><a href="{panel}"><img src="{panel}" loading="lazy"></a>'
            '<p><b>{case}</b> / {code}</p>'
            '<p><b>True label:</b> {label_name} ({grade}) &nbsp; '
            '<b>Image prediction:</b> {image_pred} (p={ip:.4f}) &nbsp; '
            '<b>Case prediction:</b> {case_pred} (mean p={cp:.4f}) &nbsp; '
            '<a href="{overlay}">standalone overlay</a></p></article>'.format(
                panel=html.escape(r["panel"]), overlay=html.escape(r["overlay"]), case=html.escape(r["case_id"]),
                code=html.escape(r["image_code"]), ip=r["probability_cin2plus"],
                cp=case_probs[r["case_key"]],
                label_name="CIN2+" if r["label"] else "CIN1-",
                image_pred="CIN2+" if r["predicted_label_at_0_5"] else "CIN1-",
                case_pred="CIN2+" if case_preds[r["case_key"]] else "CIN1-",
                grade=html.escape(r["grade"]),
            )
        )
    page = """<!doctype html><meta charset='utf-8'><title>Stage 1 F50 My_Dataset test</title>
<style>body{font-family:Arial;margin:24px;background:#f5f5f5}article{background:white;padding:12px;margin:12px 0;border-radius:8px}img{width:min(100%,1100px)}p{margin:8px 0 0}</style>
<h1>Stage 1 F50 — My_Dataset test Grad-CAM</h1>
<p><b>Exploratory:</b> three-fold checkpoint ensemble; not a protocol-final full-development refit.</p>
<p>Probability is CIN2+. Case probability is the mean across all eligible AA images. Predictions use threshold 0.5.</p>
""" + "\n".join(cards)
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

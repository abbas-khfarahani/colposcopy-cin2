#!/usr/bin/env python3
"""Grad-CAM heatmaps for test acetic images (primary ensemble encoders).

Uses the locked primary image models:
  - acetic_resnet50_stable
  - acetic_resnet50_difflr
and averages their Grad-CAMs (same 50/50 idea as the primary score ensemble).

Outputs under ml/runs/heatmaps_primary/ (never into dataset folders).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.boost import load_ckpt  # noqa: E402
from ml.data import ColposcopyAceticDataset, build_transforms  # noqa: E402
from ml.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import set_seed  # noqa: E402


class GradCAM:
    """Grad-CAM on ResNet layer4 (BinaryImageClassifier.backbone.layer4)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)
        self._bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        # binary logit → maximize positive class score
        score = logits.sum()
        score.backward()
        assert self.activations is not None and self.gradients is not None
        grads = self.gradients  # (B, C, H, W)
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        prob = float(torch.sigmoid(logits[0]).detach().cpu())
        return cam, prob


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return arr


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heat = cm.jet(cam)[:, :, :3]  # 0-1 RGB
    heat = (heat * 255).astype(np.uint8)
    out = (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)
    return out


def save_panel(rgb, cam, path: Path, title: str):
    import matplotlib.pyplot as plt

    overlay = overlay_cam(rgb, cam)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    # also save standalone overlay PNG for quick browsing
    Image.fromarray(overlay).save(path.with_name(path.stem + "_overlay.png"))


def main():
    set_seed(42)
    device = resolve_device("cuda")
    out_root = Path("ml/runs/heatmaps_primary")
    out_root.mkdir(parents=True, exist_ok=True)

    stable_p = Path("ml/runs/acetic_resnet50_stable/best.pt")
    difflr_p = Path("ml/runs/acetic_resnet50_difflr/best.pt")
    m_stable, cfg = load_ckpt(stable_p, device)
    m_difflr, _ = load_ckpt(difflr_p, device)
    m_stable.eval()
    m_difflr.eval()

    # ResNet50 last conv block
    layer_s = m_stable.backbone.layer4
    layer_d = m_difflr.backbone.layer4
    cam_s = GradCAM(m_stable, layer_s)
    cam_d = GradCAM(m_difflr, layer_d)

    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    ds = ColposcopyAceticDataset(
        split="test",
        splits_dir=cfg["splits_dir"],
        my_dataset_root=cfg["my_dataset_root"],
        transform=tf,
    )

    # person-level max scores for labeling folders
    person_scores: dict[str, list[float]] = {}
    records = []

    print(f"Generating Grad-CAM for {len(ds)} test images ...")
    for i in range(len(ds)):
        sample = ds[i]
        if len(sample) == 3:
            image, label, meta = sample
        else:
            image, label, _, meta = sample
        x = image.unsqueeze(0).to(device)
        x.requires_grad_(True)

        c1, p1 = cam_s(x)
        c2, p2 = cam_d(x)
        cam = 0.5 * c1 + 0.5 * c2
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        prob = 0.5 * p1 + 0.5 * p2

        pid = str(meta["person_id"])
        y = int(label)
        person_scores.setdefault(pid, []).append(prob)

        rgb = denormalize(image)
        safe_pid = pid.replace("/", "_")
        file_stem = Path(str(meta.get("file") or meta.get("path") or f"img{i}")).stem
        fname = f"{safe_pid}__{file_stem}"
        # temporary; relabel after person max known
        records.append(
            {
                "index": i,
                "person_id": pid,
                "y_true": y,
                "prob_ensemble": prob,
                "prob_stable": p1,
                "prob_difflr": p2,
                "fname": fname,
                "rgb": rgb,
                "cam": cam,
                "meta": meta,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)}")

    cam_s.close()
    cam_d.close()

    # person max score + outcome folder
    person_max = {pid: max(vs) for pid, vs in person_scores.items()}
    # use simple image-score threshold 0.5 for folder tags (visual triage only)
    thr = 0.5

    index_rows = []
    for r in records:
        pid = r["person_id"]
        y = r["y_true"]
        pmax = person_max[pid]
        pred = int(pmax >= thr)
        if y == 1 and pred == 1:
            tag = "TP"
        elif y == 1 and pred == 0:
            tag = "FN"
        elif y == 0 and pred == 1:
            tag = "FP"
        else:
            tag = "TN"
        sub = out_root / tag
        title = (
            f"{tag} | person={pid} | y={y} | img_p={r['prob_ensemble']:.3f} "
            f"| person_max={pmax:.3f} | thr={thr}"
        )
        panel_path = sub / f"{r['fname']}_panel.png"
        save_panel(r["rgb"], r["cam"], panel_path, title)
        index_rows.append(
            {
                "tag": tag,
                "person_id": pid,
                "y_true": y,
                "img_prob": r["prob_ensemble"],
                "person_max_prob": pmax,
                "panel": str(panel_path.relative_to(out_root)).replace("\\", "/"),
                "overlay": str(
                    panel_path.with_name(panel_path.stem + "_overlay.png")
                    .relative_to(out_root)
                ).replace("\\", "/"),
            }
        )

    summary = {
        "model": "50/50 Grad-CAM average of acetic_resnet50_stable + acetic_resnet50_difflr",
        "split": "test",
        "n_images": len(records),
        "folder_threshold_note": (
            "TP/FN/FP/TN folders use person-level max ensemble image score "
            "vs threshold 0.5 (image-only triage). Late clinical stack is not "
            "applied inside Grad-CAM."
        ),
        "counts": {
            t: sum(1 for r in index_rows if r["tag"] == t) for t in ("TP", "FN", "FP", "TN")
        },
        "items": index_rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # HTML gallery
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Grad-CAM test heatmaps (primary ensemble)</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}",
        "h1,h2{font-weight:600} a{color:#8cf}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}",
        ".card{background:#1c1c1c;border-radius:8px;padding:10px}",
        ".card img{width:100%;border-radius:4px}",
        ".meta{font-size:12px;opacity:.85;margin-top:6px}",
        ".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700}",
        ".TP{background:#1b5e20}.FN{background:#b71c1c}.FP{background:#e65100}.TN{background:#37474f}",
        "</style></head><body>",
        "<h1>Grad-CAM — test set (primary 50/50 ResNet50)</h1>",
        "<p>Heatmap = average Grad-CAM of <code>stable</code> + <code>difflr</code> "
        "(image encoders of the locked primary). Folders use person max image score @ 0.5 "
        "(not the late clinical stack).</p>",
        f"<p>Counts (images): {summary['counts']}</p>",
    ]
    for tag in ("FN", "TP", "FP", "TN"):
        rows = [r for r in index_rows if r["tag"] == tag]
        html.append(f"<h2>{tag} (n={len(rows)})</h2><div class='grid'>")
        # sort FN/TP by score ascending for FN (hardest first)
        rows = sorted(rows, key=lambda r: r["img_prob"], reverse=(tag != "FN"))
        for r in rows:
            html.append(
                "<div class='card'>"
                f"<span class='tag {tag}'>{tag}</span>"
                f"<div class='meta'>person {r['person_id']} · "
                f"img p={r['img_prob']:.3f} · person max={r['person_max_prob']:.3f}</div>"
                f"<a href='{r['panel']}'><img src='{r['overlay']}' loading='lazy'></a>"
                "</div>"
            )
        html.append("</div>")
    html.append("</body></html>")
    (out_root / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("Wrote", out_root / "index.html")
    print("Counts", summary["counts"])


if __name__ == "__main__":
    main()

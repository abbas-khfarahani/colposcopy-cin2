#!/usr/bin/env python3
"""Localization heatmaps for spatial-attention ResNet models.

Shows:
  Original | Attention map (trained) | Multi-layer Grad-CAM | Attention overlay

Outputs under ml/runs/<out_name>/ — never into dataset folders.
"""

from __future__ import annotations

import argparse
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
from ml.models.classifier import SpatialAttentionClassifier  # noqa: E402
from ml.train import resolve_device  # noqa: E402
from ml.utils import set_seed  # noqa: E402


class MultiLayerGradCAM:
    """Average Grad-CAM over several ResNet layers (e.g. layer3 + layer4)."""

    def __init__(self, model: torch.nn.Module, layers: list[torch.nn.Module]):
        self.model = model
        self.acts: list[torch.Tensor | None] = [None] * len(layers)
        self.grads: list[torch.Tensor | None] = [None] * len(layers)
        self._hooks = []
        for i, layer in enumerate(layers):

            def _fwd(module, inp, out, idx=i):
                self.acts[idx] = out

            def _bwd(module, gin, gout, idx=i):
                self.grads[idx] = gout[0]

            self._hooks.append(layer.register_forward_hook(_fwd))
            self._hooks.append(layer.register_full_backward_hook(_bwd))

    def close(self) -> None:
        for h in self._hooks:
            h.remove()

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        for i in range(len(self.acts)):
            self.acts[i] = None
            self.grads[i] = None
        out = self.model(x)
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        logits.sum().backward()
        cams = []
        for act, grad in zip(self.acts, self.grads):
            assert act is not None and grad is not None
            w = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            cams.append(cam[0, 0])
        cam = torch.stack(cams, dim=0).mean(dim=0).detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        prob = float(torch.sigmoid(logits[0]).detach().cpu())
        return cam, prob


def denormalize(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def save_panel(rgb, attn, cam, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    overlay = overlay_cam(rgb, attn)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(rgb)
    axes[0].set_title("Original")
    axes[1].imshow(attn, cmap="jet")
    axes[1].set_title("Attention")
    axes[2].imshow(cam, cmap="jet")
    axes[2].set_title("Grad-CAM (L3+L4)")
    axes[3].imshow(overlay)
    axes[3].set_title("Attn overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    Image.fromarray(overlay).save(path.with_name(path.stem + "_overlay.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="ml/runs/acetic_resnet50_attn/best.pt",
        help="Attention model checkpoint",
    )
    ap.add_argument("--out-dir", default="ml/runs/heatmaps_attn")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    set_seed(42)
    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint)
    out_root = Path(args.out_dir)
    if out_root.exists():
        for sub in ("TP", "FN", "FP", "TN"):
            p = out_root / sub
            if p.exists():
                for f in p.glob("*"):
                    f.unlink()
    out_root.mkdir(parents=True, exist_ok=True)

    model, cfg = load_ckpt(ckpt_path, device)
    if not isinstance(model, SpatialAttentionClassifier):
        raise SystemExit(
            f"Expected SpatialAttentionClassifier, got {type(model)}. "
            "Use an image_attn checkpoint."
        )
    model.eval()

    body = model.backbone
    cam_engine = MultiLayerGradCAM(model, [body.layer3, body.layer4])

    tf = build_transforms("val", int(cfg.get("image_size", 224)), False)
    ds = ColposcopyAceticDataset(
        split=args.split,
        splits_dir=cfg["splits_dir"],
        my_dataset_root=cfg["my_dataset_root"],
        transform=tf,
    )

    person_scores: dict[str, list[float]] = {}
    records = []
    print(f"Generating attention+CAM for {len(ds)} {args.split} images ...")
    for i in range(len(ds)):
        sample = ds[i]
        image, label, meta = sample[0], sample[1], sample[-1]
        x = image.unsqueeze(0).to(device)
        x.requires_grad_(True)

        with torch.enable_grad():
            logits, attn_t = model(x, return_attn=True)
            cam, _ = cam_engine(x)
        prob = float(torch.sigmoid(logits[0]).detach().cpu())
        attn = F.interpolate(attn_t, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = attn[0, 0].detach().cpu().numpy()
        attn = attn - attn.min()
        if attn.max() > 1e-8:
            attn = attn / attn.max()

        pid = str(meta["person_id"])
        y = int(label)
        person_scores.setdefault(pid, []).append(prob)
        file_stem = Path(str(meta.get("file") or meta.get("path") or f"img{i}")).stem
        fname = f"{pid.replace('/', '_')}__{file_stem}"
        records.append(
            {
                "person_id": pid,
                "y_true": y,
                "prob": prob,
                "fname": fname,
                "rgb": denormalize(image),
                "attn": attn,
                "cam": cam,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ds)}")

    cam_engine.close()
    person_max = {pid: max(vs) for pid, vs in person_scores.items()}
    thr = 0.5
    index_rows = []
    for r in records:
        pid, y, pmax = r["person_id"], r["y_true"], person_max[r["person_id"]]
        pred = int(pmax >= thr)
        if y == 1 and pred == 1:
            tag = "TP"
        elif y == 1 and pred == 0:
            tag = "FN"
        elif y == 0 and pred == 1:
            tag = "FP"
        else:
            tag = "TN"
        panel = out_root / tag / f"{r['fname']}_panel.png"
        title = (
            f"{tag} | person={pid} | y={y} | img_p={r['prob']:.3f} "
            f"| person_max={pmax:.3f} | thr={thr}"
        )
        save_panel(r["rgb"], r["attn"], r["cam"], panel, title)
        index_rows.append(
            {
                "tag": tag,
                "person_id": pid,
                "y_true": y,
                "img_prob": r["prob"],
                "person_max_prob": pmax,
                "panel": str(panel.relative_to(out_root)).replace("\\", "/"),
                "overlay": str(
                    panel.with_name(panel.stem + "_overlay.png").relative_to(out_root)
                ).replace("\\", "/"),
            }
        )

    summary = {
        "model": str(ckpt_path),
        "model_type": cfg.get("model_type"),
        "image_size": cfg.get("image_size"),
        "split": args.split,
        "n_images": len(records),
        "counts": {
            t: sum(1 for r in index_rows if r["tag"] == t) for t in ("TP", "FN", "FP", "TN")
        },
        "items": index_rows,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Attention heatmaps</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}",
        ".card{background:#1c1c1c;border-radius:8px;padding:10px}.card img{width:100%}",
        ".tag{display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700}",
        ".TP{background:#1b5e20}.FN{background:#b71c1c}.FP{background:#e65100}.TN{background:#37474f}",
        "</style></head><body>",
        f"<h1>Attention localization — {ckpt_path.parent.name}</h1>",
        f"<p>image_size={cfg.get('image_size')} · counts={summary['counts']}</p>",
    ]
    for tag in ("FN", "TP", "FP", "TN"):
        rows = [r for r in index_rows if r["tag"] == tag]
        rows = sorted(rows, key=lambda r: r["img_prob"], reverse=(tag != "FN"))
        html.append(f"<h2>{tag} (n={len(rows)})</h2><div class='grid'>")
        for r in rows:
            html.append(
                "<div class='card'>"
                f"<span class='tag {tag}'>{tag}</span>"
                f"<div>person {r['person_id']} · p={r['img_prob']:.3f} · max={r['person_max_prob']:.3f}</div>"
                f"<a href='{r['panel']}'><img src='{r['overlay']}' loading='lazy'></a></div>"
            )
        html.append("</div>")
    html.append("</body></html>")
    (out_root / "index.html").write_text("\n".join(html), encoding="utf-8")
    print("Wrote", out_root / "index.html")
    print("Counts", summary["counts"])


if __name__ == "__main__":
    main()

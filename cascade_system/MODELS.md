# Locked five-model screen (M1–M5)

Fixed **architectures** for every cascade stage.  
Cascade models for S1+ are chosen after S0 under this system — not by cherry-picking after S1–S3.

| ID | Model | Input size | Role |
|---|---|---|---|
| **M1** | ResNet50 + GAP | 224 | **Primary** — classical CNN baseline |
| **M2** | ResNet50 + GAP | 384 | Same backbone, higher resolution |
| **M3** | EfficientNet-B3 | 300 | Compound-scaled modern CNN |
| **M4** | ConvNeXt-Tiny | 224 | Modernized CNN (Transformer-style design) |
| **M5** | ResNet50 + spatial attention | 224 | Localization head — “does cascade reduce need for attention?” |

## Legacy config references (exploratory / old runs)

These JSON files in the main repo are **recipes to mirror**, not automatic result imports:

| ID | Config hint |
|---|---|
| M1 | `ml/configs/acetic_resnet50_stable.json` |
| M2 | `ml/configs/acetic_resnet50_stable_384.json` |
| M3 | `ml/configs/acetic_efficientnet_b3_stable.json` |
| M4 | `ml/configs/acetic_convnext_tiny_stable.json` |
| M5 | `ml/configs/acetic_resnet50_attn.json` |

Protocol cousins (person OOF style): B1 ≈ M1, B2 ≈ M4, B4 ≈ M5 — useful for methodology, not for copying old AUROCs into this system’s tables.

## Reporting rules

- Screen / report **all five** on S0 (or pre-declare primary + diversity before cascade).  
- Do not drop a model after seeing S1–S3 only because cascade helped it less.  
- Main claim: modules improve / stabilize detection **across** architectures.  
- Secondary: which backbone is strongest *after* clean inputs.

## Result directories

Each stage has:

```text
results/<STAGE>/M1/
results/<STAGE>/M2/
results/<STAGE>/M3/
results/<STAGE>/M4/
results/<STAGE>/M5/
```

Expected artifacts per run (when executed later): `summary.json`, OOF predictions, optional heatmaps index — details filled when training starts.

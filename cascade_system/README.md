# Cascade system — modular CIN2+ colposcopy CAD

Standalone workspace for the **multi-module** approach (quality / ROI cascade + diagnostic CNNs).  
Kept separate from legacy `ml/protocol` and exploratory `ml/runs` so tracking stays clean.

This folder is for **design, step status, and paper-ready result slots**.  
Training code and large images stay linked from the main project; they are not duplicated here.

## Locked scientific order

```text
1) S0            Train M1–M5 on ORIGINAL acetic frames          ← true baseline
2) S1-oracle     Same protocol on HAND CROPS                    ← crop upper bound
3) S1-classical  Automatic GrabCut → bbox crop
4) S1-learned    Automatic cropper trained on hand boxes (if needed)
5) Compare       S0 vs S1-classical vs S1-learned vs S1-oracle
6) S2 → S3 …     Only on the chosen *automatic* S1 path
```

**Important:** Older five-model scores that were run on hand crops are a **temporary reference only**.  
They are **not** final S0 and **not** final S1-oracle until re-run under this system’s locked protocol.

## Folder map

| Path | Role |
|---|---|
| `FLOWCHART.md` | System flowchart (modules + data flow) |
| `PLAN.md` | Locked decisions, acceptance rules, rerun policy |
| `MODELS.md` | M1–M5 definitions |
| `STATUS.md` | Step-by-step checklist (update as you go) |
| `data/` | Pointers / mounts for originals & hand crops (TBD paths) |
| `results/` | Empty slots per stage × model for metrics & logs |
| `paper/` | Tables / figures exported for the manuscript |
| `configs/` | Future run configs (per stage / model) |
| `logs/` | High-level run logs |

## How to use when the dataset grows

1. Update `data/README.md` paths (or remount new cohorts).  
2. Re-run from the stage you need (`S0` first if labels/splits change).  
3. Write outputs into the matching `results/<STAGE>/<MODEL>/` slot.  
4. Refresh `results/comparisons/` and `STATUS.md`.  
5. Copy paper numbers into `paper/tables/`.

## Related existing project pieces (read-only references)

| Need | Existing location |
|---|---|
| Classical cervix ROI | `ml/scripts/demo_auto_lesion_masks.py` (`cervix_roi`) |
| Person OOF protocol | `ml/protocol/` |
| Legacy holdout configs | `ml/configs/acetic_*.json` |
| Prior ROI side note (mask-fill hurt) | `ml/runs/cervix_roi_side/` |

Do not mix new cascade claims into frozen protocol B0–B12 without an explicit paper decision.

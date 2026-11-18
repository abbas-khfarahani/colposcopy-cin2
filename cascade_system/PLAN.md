# Locked plan

## Scope

Modular cascade for person-level **CIN2+** detection on acetic colposcopy images.  
Diagnostic CNNs are one stage; ROI / quality / artifact modules come first.

## Cohort & splits (target)

| Item | Choice |
|---|---|
| Persons | BX1-gold style cohort (~492); grows later |
| Primary metric | Person-level OOF AUROC (person-grouped folds) |
| Aggregation | Max over acetic frames that survive gates (abstain rules logged) |
| Images | **Originals** for S0 / automatic S1; **hand crops** for S1-oracle & learned labels |

Paths: see `data/README.md` (TBD until you mount cleaned trees).

## Five models (architectures only)

See `MODELS.md`. Same five for every stage so ablations isolate **modules**, not backbone swaps.

## Stage definitions

| Stage | Input | Purpose | Deployable? |
|---|---|---|---|
| **S0** | Original frames | Baseline image-only CNN | Yes (baseline) |
| **S1-classical** | Original → GrabCut bbox crop | Auto remove wall/tools | Yes |
| **S1-learned** | Original → learned crop | Close gap to hand crop if classical weak | Yes (optional) |
| **S1-oracle** | Hand crops | Upper bound for “perfect crop” | **No** — analysis only |
| **S2** | Automatic S1 output | Blur / unusable gate | Yes |
| **S3** | S2-kept frames | Tight cervix mask + glare/artifacts | Yes |
| **S4** (optional) | Image score + clinical | Late fusion | Yes |

**Rule:** Do not build the main S2/S3 system claim on oracle crops. Oracle is a ceiling for the crop question.

## Legacy numbers

Any previous M1–M5 metrics computed on hand crops:

- Keep only as **informal reference** in notes if useful  
- **Re-run** under this folder’s protocol for official `results/S0_original` and `results/S1_oracle`  
- Do not paste old scores into `paper/tables/` as final

## S1 method notes

- Default automatic crop: **bbox crop** (not outside-fill). Prior mask-fill side test hurt AUC.  
- Fallback when ROI fails: mild center crop or full frame + `roi_failed` flag (no silent person drop).  
- Hand crops: evaluate classical/learned with box IoU; run oracle arm for CIN2+ ceiling.

## Rerun policy (growing dataset)

1. Freeze a `DATA_CARD` snapshot (person list, split seed, image roots) under `logs/` or `results/comparisons/`.  
2. Re-run affected stages from S0 if membership/labels change.  
3. Never overwrite; write `run_<timestamp>/` inside the stage slot or bump a run id in `STATUS.md`.

## Success criteria (S1 block)

| Check | Pass idea |
|---|---|
| Visual QA | Crops look like hand examples on most frames |
| Crop IoU | Classical/learned vs hand box reported |
| Metrics | Person AUROC + Δ vs S0 for all filled arms |
| Interpretation | Use oracle decision tree in `FLOWCHART.md` |

## Deferred / parked

- Grad-CAM-guided attention (M1→M5) + M1/M5 ensemble — **after** cascade S1–S3  
- Clinical fusion S4 — after image cascade is stable  
- Mask-fill as default S1 — rejected unless a controlled ablation says otherwise

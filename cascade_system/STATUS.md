# Status tracker

Update this file as stages complete. Do not mark a stage done without files under `results/`.

## Data readiness

| Item | Status | Notes |
|---|---|---|
| Original acetic tree mounted | **TBD** | See `data/README.md` |
| Hand-crop tree (all 492) mounted | **TBD** | Cleaned crop-only folder coming |
| Person splits locked for this system | **TBD** | Prefer same person OOF spirit as protocol |
| DATA_CARD snapshot | Not started | |

## Stage checklist

| Stage | Status | Results path | Paper-ready? |
|---|---|---|---|
| Design / flowchart / model lock | **Done** | this folder | Yes (methods) |
| S0 originals × M1–M5 | Not started | `results/S0_original/` | No |
| S1-oracle hand crops × M1–M5 | Not started | `results/S1_oracle/` | No |
| S1-classical × M1–M5 | Not started | `results/S1_classical/` | No |
| S1-learned × M1–M5 | Not started | `results/S1_learned/` | No |
| S1 comparison table | Not started | `results/comparisons/` | No |
| S2 quality | Blocked on automatic S1 | `results/S2_quality/` | No |
| S3 mask / artifacts | Blocked on S2 | `results/S3_artifact/` | No |
| Parked: Grad-CAM-guided M5 + ensemble | Later | — | — |

## Current next action

1. Add cleaned **hand-crop** tree and **original** tree; update `data/README.md` paths.  
2. Lock split / DATA_CARD for this system.  
3. Run **S0 on originals** for M1–M5 into `results/S0_original/`.  
4. Re-run **S1-oracle** under the same protocol into `results/S1_oracle/`.

## Legacy note

Prior five-model numbers on hand crops = **reference only** until S0 + S1-oracle are re-run here.

# Cascade system flowchart

## End-to-end modules

```mermaid
flowchart TD
  RAW[Original acetic frames] --> S0
  HAND[Hand crops — all 492 persons] --> S1O

  subgraph S0["S0 — Baseline on originals"]
    M0[M1–M5 train + person OOF]
  end

  RAW --> S1C
  subgraph S1C["S1-classical — auto crop"]
    GC[GrabCut cervix ROI]
    BB[BBox + margin crop]
    FB[Fallback + roi_failed flag]
    GC --> BB
    GC --> FB
  end

  HAND --> S1L
  RAW --> S1L
  subgraph S1L["S1-learned — optional"]
    DET[Crop model trained on hand boxes]
  end

  subgraph S1O["S1-oracle — upper bound"]
    OR[Hand crop windows]
  end

  S1C --> CMP[Compare crop arms]
  S1L --> CMP
  S1O --> CMP
  S0 --> CMP

  CMP --> PICK{Choose automatic S1<br/>classical or learned}
  PICK --> S2

  subgraph S2["S2 — Quality gate"]
    BLUR[Blur / exposure / occlusion]
    OK[Keep or down-weight]
    REJ[Reject / abstain]
    BLUR --> OK
    BLUR --> REJ
  end

  OK --> S3
  subgraph S3["S3 — Tight mask + artifacts"]
    MASK[True cervix mask]
    GL[Specular / glare cleanup]
    MASK --> GL
  end

  GL --> CNN[Same M1–M5 diagnostic heads]
  CNN --> AGG[Person aggregation]
  AGG --> OUT[CIN2+ score + optional S4 clinical fusion]
```

## S1 decision tree (uses oracle)

```mermaid
flowchart LR
  A[S0 on originals] --> B[S1-oracle on hand crops]
  B --> C{oracle ≫ S0?}
  C -->|No| D[Crop not main bottleneck → prioritize S2/S3]
  C -->|Yes| E[Run S1-classical]
  E --> F{classical ≈ oracle?}
  F -->|Yes| G[Keep classical as system S1]
  F -->|No| H[Train S1-learned]
  H --> I{learned closes gap?}
  I -->|Yes| J[System S1 = learned]
  I -->|No| K[Document ceiling; still improve classical/labels]
  G --> L[Build S2 on automatic S1 only]
  J --> L
```

## Result slot flow (for paper tracking)

```text
results/S0_original/{M1..M5}/
results/S1_oracle/{M1..M5}/
results/S1_classical/{M1..M5}/
results/S1_learned/{M1..M5}/   ← fill if trained
results/comparisons/           ← ΔAUROC tables, IoU vs hand box
results/S2_quality/...
results/S3_artifact/...
        ↓
paper/tables/  paper/figures/
```

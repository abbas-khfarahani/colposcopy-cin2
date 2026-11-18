# Orginal Data cleanup & review tools

Local Flask galleries used to clean and re-label images under `Orginal Data/` before mounting into the rest of the project (e.g. `cascade_system/data`).

## Layout

```
tools/original_data_cleanup/
  review_*_images.py   # one gallery per modality / leftover set
  _paths.py            # shared ROOT / DATA / logs / cache paths
  logs/                # rename/delete audit trails + resume state
    reports/           # one-off audit lists from the cleanup pass
  cache/               # JPEG preview cache (safe to delete; regenerates)
```

Data itself stays in the project root folder `Orginal Data/` (spelling kept as-is).

## Requirements

```bash
pip install flask pillow
```

Run from the **project root** (`bachelor Project/`).

## Reviewers

| Script | What it shows | Port | Actions |
|--------|---------------|------|---------|
| `review_d_images.py` | `_D` images | 8765 | → A/G/L/Z or Delete |
| `review_all_images.py` | all images | 8766 | Delete |
| `review_a_images.py` | `_A` images | 8767 | → G/L/Z or Delete |
| `review_l_images.py` | `_L` images | 8768 | → A/G/Z or Delete |
| `review_g_images.py` | `_G` images | 8769 | → A/L/Z or Delete |
| `review_z_images.py` | `_Z` (zoom) | 8770 | → A/G/L or Delete |
| `review_other_images.py` | not A/L/G/Z (e.g. charts) | 8771 | → A/G/L/Z or Delete |

Example:

```bash
python tools/original_data_cleanup/review_a_images.py
# browser → http://127.0.0.1:8767
```

Leave correct images alone. Wrong ones: rename with the letter buttons, or Delete.

## Naming when you rename

- Destination letter gets the **lowest free** number (`A1`+`A3` → new file becomes `A2`).
- Source letter is **not** renumbered (gaps like `A1`, `A3` are OK).

## Logs

Each action appends a JSON line under `logs/` (e.g. `a_review_log.jsonl`).  
`logs/reports/` holds older inventory/audit text from the cleanup.

## Cache

Preview JPEGs live under `cache/`. Delete that folder anytime to free disk; the next run rebuilds it.

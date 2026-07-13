# Master Funnel — Safety-First Image QC & Dataset-Prep Pipeline

A command-line tool that helps photographers turn large batches of raw images into
clean, curated computer-vision datasets. It scores image quality, removes near-duplicates,
tiers keepers, and organizes everything into folders — **without deleting anything
automatically.**

Built by **Golam Rob (PMP)®** · [golamrob.com](https://www.golamrob.com)

---

## Why "safety-first"?

Curation tools that auto-delete are dangerous — one bad threshold and you lose good photos.
Master Funnel is built the opposite way:

- **Nothing is deleted.** Files are *moved* into category folders; you review, then delete manually.
- **Weak images go to `Review/`, not `Reject/`.** Only unreadable/corrupt files are auto-rejected.
- **`--calibrate` mode** prints metric percentiles so you set thresholds from *your* data before touching anything.
- **`--dry-run`** shows exactly where each file would go, moving nothing.

## Pipeline (5 layers)

`Extractor → Deduplicator → Classifier → Mover → Reporter`

1. **Extractor** — per-image sharpness, brightness, noise, and a perceptual hash (pHash); optional AI aesthetic score (SigLIP).
2. **Deduplicator** — connected-components (union-find) grouping: transitive and **order-independent**; keeps the best frame per group (sharpness with a noise penalty).
3. **Classifier** — weighted K-Means tiers (Outstanding / Excellent / Good / Average). Tiers are a heuristic, so raw scores are also written to the audit CSV.
4. **Mover** — moves files into category folders; optional derivative crops (16:9 / 1:1 / 3:4) for keepers, with correct EXIF (no stale dimensions).
5. **Reporter** — writes a full audit CSV of every file, its metrics, and where it went.

## Install

```bash
pip install -r requirements.txt
```

`opencv-python` is required. `torch` + `transformers` are optional — only needed for the
AI aesthetic score. Deduplication, QA, and `--calibrate` work without them.

## Usage

**Always calibrate first, then dry-run, then inspect — before deleting anything:**

```bash
# 1. See your metric distribution (moves nothing, no model needed)
python master_funnel_v6.py --calibrate --source "/path/to/photos"

# 2. Preview where files would go (moves nothing)
python master_funnel_v6.py --dry-run  --source "/path/to/photos"

# 3. Run for real (still only MOVES into folders; you delete manually after review)
python master_funnel_v6.py --source "/path/to/photos" --model "/path/to/SigLIP_Model"
```

Multiple source folders are supported (cross-folder de-duplication):

```bash
python master_funnel_v6.py --source "/photos/50mm" "/photos/18-140mm" --output "/photos/_FUNNEL"
```

| Flag | Purpose |
|---|---|
| `--source` | One or more input folders (space-separated) |
| `--output` | Output root (default: first source folder) |
| `--model` | Local SigLIP model folder (only for AI scoring) |
| `--config` | JSON config to override thresholds/prompts |
| `--dry-run` | Report only — move nothing |
| `--calibrate` | Print metric percentiles — move nothing, no model needed |

## Output folders

`Outstanding/ Excellent/ Good/ Average/` (keepers) · `Review/` (manual check) ·
`Reject/` (unreadable only) · `Duplicates/` · `Quarantine/` (fallback on move errors)

## Notes

- Configuration (thresholds, prompts, crop settings) lives in `DEFAULT_CONFIG` at the top of the script, overridable via `--config`.
- Default thresholds are a starting point — validate them on your own images with `--calibrate`.

## License

MIT — free to use and adapt. Attribution appreciated: *Golam Rob — golamrob.com*.

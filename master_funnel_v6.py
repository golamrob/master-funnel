"""
Master Funnel Pipeline v6.0 — Safety-First AI Dataset Curator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v6.0 — রিভিউ-কমেন্ট অনুযায়ী সব ফিক্স (v5.0 থেকে)

🔴 CRITICAL FIXES
  ✔ [PROMPT]   positive_prompt "water lilies" ছিল — অন্য প্রজেক্টের copy-paste।
               এখন ম্যানগ্রোভ/মোহনা/waterline অনুযায়ী। (AI score আর অর্থহীন নয়)
  ✔ [SIGLIP]   _ai_score_batch text broadcast বাগ ([pos,neg]*N → [pos,neg])।
  ✔ [CROP]     crop আগে DUPLICATE-এ হতো (উল্টো) — এখন KEEPER-এ, এবং opt-in।
  ✔ [CROP-EXIF] crop-এ আসল EXIF dimension/thumbnail কপি হতো → mismatch (v4 বাগ)।
               এখন stale ImageWidth/Height/PixelDimension/thumbnail বাদ দিয়ে কপি।
  ✔ [NAMING]   crop suffix "8/88/888" (BDC_00018 বিভ্রান্তি) → "_16x9/_1x1/_3x4"।

🟠 SAFETY / CORRECTNESS
  ✔ [SAFETY]   threshold-fail এখন auto-Reject নয় → "Review" (ম্যানুয়াল চেক)।
               শুধু unreadable/corrupt ফাইল auto-Reject। ভালো নরম ফ্রেম হারাবে না।
  ✔ [SAFETY]   silent "or Reject" fallback → এখন "Review" + WARNING।
  ✔ [CALIBRATE] নতুন --calibrate মোড: metric percentile ছাপে, কিছু move করে না।
               ডিলিটের আগে threshold ঠিক করার জন্য।
  ✔ [DEDUP]    connected-components (union-find) → transitive ও order-independent।
  ✔ [DEDUP]    keeper = sharpness − noise_penalty×noise (noise-inflated sharp নয়)।
  ✔ [MULTI]    --source একাধিক ফোল্ডার নেয় → cross-folder (৫০mm+১৮-১৪০mm) dedup।
  ✔ [RANK]     K-Means এখন weighted [ai_score×2, sharpness] (edge-energy double
               count নয়); tier heuristic, তাই CSV-তে raw score-ও থাকে।

⚙️ ARCH (v5 থেকে রাখা)
  ✔ ৫ লেয়ার: Extractor → Deduplicator → Classifier → Mover → Reporter
  ✔ ImageRecord dataclass, context manager (no leak), quarantine fallback
  ✔ heavy import (torch/transformers/cv2) lazy → দ্রুত startup, calibrate-এ model লাগে না
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  ডিলিটের আগে সবসময়:  python master_funnel_v6.py --calibrate --source <folder>
    তারপর:              python master_funnel_v6.py --dry-run  --source <folder>
    Reject/Review/Duplicates ফোল্ডার চোখে দেখে তবেই ম্যানুয়ালি ডিলিট।
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import imagehash
import piexif
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# NOTE: torch / transformers / cv2 lazily imported inside the functions that need
# them, so --calibrate and pure-logic runs don't require the ML stack or GPU.


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ১. কনফিগারেশন
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CONFIG: dict = {
    # ── Technical gates ─ VALIDATE with --calibrate before trusting! ─────────
    "min_sharpness": 8.0,
    "min_brightness": 25.0,
    "max_brightness": 250.0,        # overexposure guard (new)
    "max_noise_sigma": 18.0,
    # threshold-fail কে কোথায় পাঠাবে: True → "Review" (নিরাপদ, ম্যানুয়াল),
    # False → "Reject" (আক্রমণাত্মক)। unreadable ফাইল সবসময় Reject।
    "soft_fail_to_review": True,

    # ── Deduplication ───────────────────────────────────────────────────────
    "hash_distance_threshold": 6,   # ছোট = কড়া। ধীর নৌকায় 8–10 লাগতে পারে।
    "keeper_noise_penalty": 0.5,    # keeper_score = sharpness − penalty × noise_sigma

    # ── AI aesthetic scoring (SigLIP zero-shot) ─────────────────────────────
    # 🔴 FIXED: আগে "blooming water lilies" ছিল — সম্পূর্ণ ভুল ডোমেইন।
    "positive_prompt": (
        "a high-quality documentary photograph of a mangrove river estuary at "
        "high tide, sharp focus, clear waterline between water and vegetation, "
        "natural detail, well composed, editorial quality"
    ),
    "negative_prompt": (
        "a low quality, blurry, out-of-focus, noisy, badly exposed, tilted, "
        "featureless amateur snapshot"
    ),

    # ── Foldering ───────────────────────────────────────────────────────────
    # ASCII folder names (path/encoding-safe). Review = manual check bucket.
    "categories": ["Outstanding", "Excellent", "Good", "Average",
                   "Review", "Reject", "Duplicates"],

    # ── Optional derivative crops (marketing aspect ratios) ─────────────────
    "generate_crops": False,        # opt-in. crops KEEPER-দের জন্য (fixed)।
    "crop_categories": ["Outstanding", "Excellent", "Good", "Average"],
    "crop_versions": [
        {"suffix": "_16x9", "aspect": [16, 9]},
        {"suffix": "_1x1",  "aspect": [1, 1]},
        {"suffix": "_3x4",  "aspect": [3, 4]},
    ],
    "crop_center": True,            # True → center crop, False → rule-of-thirds

    # ── Inference ───────────────────────────────────────────────────────────
    "ai_resize_max_px": 512,
    "batch_size": 8,
}

RANK_LABELS = ["Outstanding", "Excellent", "Good", "Average"]
DUP_CAT = "Duplicates"
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")


def load_config(config_path: Optional[str]) -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Master Funnel v6.0 — safety-first curator")
    # 🟠 MULTI: একাধিক ফোল্ডার → cross-folder dedup
    p.add_argument("--source", nargs="+", required=True,
                   help="এক বা একাধিক সোর্স ফোল্ডার (স্পেস দিয়ে আলাদা)")
    p.add_argument("--output", default=None,
                   help="আউটপুট রুট (default: প্রথম সোর্স ফোল্ডার)")
    p.add_argument("--model", default=r"C:\Users\GolamRob\SigLIP_Model",
                   help="লোকাল SigLIP মডেল ফোল্ডার")
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="কিছু move করবে না — শুধু কোনটা কোথায় যেত তা রিপোর্ট")
    p.add_argument("--calibrate", action="store_true",
                   help="শুধু metric percentile ছাপে (model/move ছাড়া) — threshold ঠিক করতে")
    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ২. ডেটা মডেল
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ImageRecord:
    file: str
    path: str
    source: str = ""
    sharpness: float = 0.0
    brightness: float = 0.0
    composition: float = 0.0          # informational only (edge energy) — ranking-এ নয়
    noise_sigma: float = 0.0
    ai_score: float = 50.0
    phash: Optional[object] = field(default=None, repr=False)
    category: Optional[str] = None
    reject_reason: Optional[str] = None
    is_best_of_group: bool = False
    group_id: int = -1
    final_path: Optional[str] = None

    @property
    def keeper_score(self) -> float:
        """dedup-এ কোনটা সেরা — noise-penalised sharpness।"""
        pen = float(DEFAULT_CONFIG["keeper_noise_penalty"])
        return self.sharpness - pen * self.noise_sigma


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৩. ইনফ্রাস্ট্রাকচার
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_logging(out_dir: Path) -> logging.Logger:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"funnel_v6_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler()],
    )
    return logging.getLogger("funnel")


def detect_device():
    """AMD DirectML → CUDA → CPU (torch lazy import)।"""
    import torch
    try:
        import torch_directml
        logging.info("Using DirectML (AMD GPU)")
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        logging.info("Using CUDA (NVIDIA GPU)")
        return torch.device("cuda")
    logging.info("Using CPU")
    return torch.device("cpu")


def load_ai_model(model_path: str, device):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    model.eval()
    return processor, model


def collect_images(sources: list[Path], logger: logging.Logger) -> list[tuple[str, str]]:
    """সব সোর্স থেকে (source, path) — top-level only (category ফোল্ডার এড়াতে)।"""
    skip = set(DEFAULT_CONFIG["categories"]) | {"Quarantine", "logs"}
    out: list[tuple[str, str]] = []
    for src in sources:
        if not src.exists():
            logger.warning("সোর্স নেই, বাদ: %s", src); continue
        for name in os.listdir(src):
            full = src / name
            if full.is_file() and name.lower().endswith(IMG_EXT):
                out.append((str(src), str(full)))
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৪. LAYER A — Extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analyze_technical(img: Image.Image) -> dict:
    """শুধু metric — কোনো I/O নেই (cv2 lazy import)।"""
    import cv2
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = lap_var * ((h * w) / 1_000_000) ** 0.3
    brightness = float(np.mean(gray))

    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    noise_sigma = float(np.std(gray.astype(np.float32) - blurred))

    # informational edge energy near rule-of-thirds points (ranking-এ ব্যবহার হয় না)
    pts = [(int(h/3), int(w/3)), (int(h/3), int(w*2/3)),
           (int(h*2/3), int(w/3)), (int(h*2/3), int(w*2/3))]
    comp = 0.0
    for y, x in pts:
        patch = gray[max(0, y-60):min(h, y+60), max(0, x-60):min(w, x+60)]
        if patch.size:
            comp += cv2.Laplacian(patch, cv2.CV_64F).var()
    return {"sharpness": sharpness, "brightness": brightness,
            "composition": comp / 4.0, "noise_sigma": noise_sigma}


def _resize_for_ai(img: Image.Image, max_px: int) -> Image.Image:
    w, h = img.size
    if max(w, h) > max_px:
        r = max_px / max(w, h)
        return img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    return img


def _ai_score_batch(images, processor, model, device,
                    positive_prompt: str, negative_prompt: str) -> list[float]:
    """
    SigLIP zero-shot batch scoring।
    🔴 FIXED: text ছিল [pos,neg]*N (broadcast বাগ) → এখন ঠিক দুটি text।
    score = sigmoid(positive logit) × 100  (SigLIP pairwise sigmoid)।
    """
    import torch
    try:
        inputs = processor(
            text=[positive_prompt, negative_prompt],   # exactly 2 texts
            images=images, padding="max_length", return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image        # [N, 2]
            probs = torch.sigmoid(logits)                    # SigLIP: per-pair sigmoid
        return [float(probs[i][0].item() * 100) for i in range(len(images))]
    except Exception as e:
        logging.warning("Batch AI scoring failed → 50.0: %s", e)
        return [50.0] * len(images)


def _flush_ai_batch(buf, records, processor, model, device, config) -> None:
    indices, imgs = zip(*buf)
    scores = _ai_score_batch(list(imgs), processor, model, device,
                             config["positive_prompt"], config["negative_prompt"])
    for idx, sc in zip(indices, scores):
        records[idx].ai_score = sc


def _technical_verdict(rec: ImageRecord, config: dict) -> tuple[bool, Optional[str]]:
    """(ok, reason)। ok=False → keeper নয়।"""
    reasons = []
    if rec.sharpness < config["min_sharpness"]:
        reasons.append("low-sharpness")
    if rec.brightness < config["min_brightness"]:
        reasons.append("too-dark")
    if rec.brightness > config["max_brightness"]:
        reasons.append("too-bright")
    if rec.noise_sigma > config["max_noise_sigma"]:
        reasons.append("noisy")
    if reasons:
        return False, ", ".join(reasons)
    return True, None


def extract_all(image_pairs, processor, model, device, config,
                score_ai: bool = True) -> list[ImageRecord]:
    """
    LAYER A: metric + pHash + (optional) AI score।
    🟠 SAFETY: threshold-fail → "Review" (default), unreadable → "Reject"।
    score_ai=False (calibrate) হলে model ছাড়াই চলে।
    """
    logger = logging.getLogger("funnel")
    records: list[ImageRecord] = []
    buf: list[tuple[int, Image.Image]] = []
    bs = config.get("batch_size", 8)
    max_px = config["ai_resize_max_px"]
    soft_dest = "Review" if config.get("soft_fail_to_review", True) else "Reject"

    for source, path in tqdm(image_pairs, desc="Extracting"):
        rec = ImageRecord(file=os.path.basename(path), path=path, source=source)
        records.append(rec)
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                t = _analyze_technical(rgb)
                rec.sharpness, rec.brightness = t["sharpness"], t["brightness"]
                rec.composition, rec.noise_sigma = t["composition"], t["noise_sigma"]
                try:
                    rec.phash = imagehash.phash(rgb)
                except Exception:
                    rec.phash = None

                ok, reason = _technical_verdict(rec, config)
                if not ok:
                    rec.category = soft_dest      # Review (safe) — auto-delete নয়
                    rec.reject_reason = reason
                elif score_ai:
                    buf.append((len(records) - 1, _resize_for_ai(rgb.copy(), max_px)))
        except Exception as e:
            logger.error("পড়া গেল না [%s]: %s", rec.file, e)
            rec.category = "Reject"               # corrupt/unreadable → Reject
            rec.reject_reason = f"unreadable: {e}"

        if score_ai and len(buf) >= bs:
            _flush_ai_batch(buf, records, processor, model, device, config); buf.clear()

    if score_ai and buf:
        _flush_ai_batch(buf, records, processor, model, device, config)
    return records


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৫. LAYER B — Deduplicator  (connected components / union-find)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _DSU:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def deduplicate(records: list[ImageRecord], threshold: int
                ) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """
    🟠 FIXED: transitive (A~B, B~C ⇒ একই গ্রুপ) ও order-independent।
    প্রতি গ্রুপে keeper = সর্বোচ্চ keeper_score (noise-penalised); tie → filename।
    """
    valid = [r for r in records if r.category is None]
    n = len(valid)
    keepers, dups = [], []
    if n == 0:
        return keepers, dups

    dsu = _DSU(n)
    idx_with_hash = [i for i in range(n) if valid[i].phash is not None]
    for a in range(len(idx_with_hash)):
        i = idx_with_hash[a]
        for b in range(a + 1, len(idx_with_hash)):
            j = idx_with_hash[b]
            if (valid[i].phash - valid[j].phash) <= threshold:
                dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)

    gid = 0
    for root in sorted(groups):
        members = groups[root]
        gid += 1
        # deterministic keeper: score desc, তারপর filename asc
        members.sort(key=lambda k: (-valid[k].keeper_score, valid[k].file))
        best = members[0]
        valid[best].is_best_of_group = True
        valid[best].group_id = gid
        keepers.append(valid[best])
        for k in members[1:]:
            valid[k].category = DUP_CAT
            valid[k].reject_reason = "near-duplicate"
            valid[k].group_id = gid
            dups.append(valid[k])
    return keepers, dups


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৬. LAYER C — Classifier  (weighted K-Means tiers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify(keepers: list[ImageRecord]) -> None:
    """
    🟡 FIXED: ranking primary = ai_score (এখন অর্থপূর্ণ), secondary = sharpness।
    edge-energy 'composition' ranking থেকে বাদ (sharpness-এর সাথে correlated)।
    tier heuristic মাত্র — raw score CSV-তে থাকে।
    """
    n = len(keepers)
    if n == 0:
        return
    if n < 2:
        keepers[0].category = "Good"
        return

    ai = np.array([r.ai_score for r in keepers], dtype=np.float64)
    sh = np.array([r.sharpness for r in keepers], dtype=np.float64)
    feats = np.column_stack([np.log1p(ai), np.log1p(sh)])
    scaled = MinMaxScaler().fit_transform(feats)
    scaled *= np.array([2.0, 1.0])                 # ai_score-কে বেশি ওজন

    k = min(len(RANK_LABELS), n)
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(scaled)

    composite = scaled.sum(axis=1)
    means = {c: float(composite[labels == c].mean()) for c in range(k)}
    order = sorted(means, key=means.get, reverse=True)
    rank_map = {c: (RANK_LABELS[i] if i < len(RANK_LABELS) else "Average")
                for i, c in enumerate(order)}
    for rec, lbl in zip(keepers, labels):
        rec.category = rank_map[lbl]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৭. LAYER D — Mover
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_move(src: str, dst: str, quarantine_dir: str) -> Optional[str]:
    try:
        if not Path(dst).exists():
            shutil.move(src, dst); return dst
        base, ext = os.path.splitext(dst)
        c = 1
        while Path(f"{base}-{c}{ext}").exists():
            c += 1
        new = f"{base}-{c}{ext}"
        shutil.move(src, new); return new
    except Exception as e:
        logging.error("মুভ ব্যর্থ [%s]: %s → quarantine", src, e)
        try:
            Path(quarantine_dir).mkdir(parents=True, exist_ok=True)
            shutil.move(src, os.path.join(quarantine_dir, Path(src).name))
        except Exception as qe:
            logging.critical("Quarantine-ও ব্যর্থ [%s]: %s", src, qe)
        return None


def _copy_exif_no_dims(src: str, dst: str) -> bool:
    """
    🔴 FIXED: crop-এ আসল EXIF কপি করলে stale ImageWidth/Height/thumbnail বসে
    (v4 mismatch বাগ)। এখানে dimension + thumbnail বাদ দিয়ে বাকি tag কপি।
    """
    try:
        with Image.open(src) as im:
            exif_bytes = im.info.get("exif")
        if exif_bytes:
            d = piexif.load(exif_bytes)
            for t in (piexif.ImageIFD.ImageWidth, piexif.ImageIFD.ImageLength):
                d.get("0th", {}).pop(t, None)
            for t in (piexif.ExifIFD.PixelXDimension, piexif.ExifIFD.PixelYDimension):
                d.get("Exif", {}).pop(t, None)
            d["1st"] = {}
            d["thumbnail"] = None            # stale thumbnail বাদ
            piexif.insert(piexif.dump(d), dst)
            return True
    except Exception as e:
        logging.debug("piexif crop-exif ব্যর্থ [%s]: %s", dst, e)
    # ExifTool fallback — dimension/thumbnail বাদ দিয়ে
    try:
        subprocess.run(
            ["exiftool", "-overwrite_original", "-tagsFromFile", src, "-all:all",
             "--EXIF:ImageWidth", "--EXIF:ImageHeight",
             "--EXIF:ExifImageWidth", "--EXIF:ExifImageHeight", "-IFD1:all=", dst],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:
        logging.error("exiftool crop-exif ব্যর্থ [%s]: %s", dst, e)
    return False


def _generate_crops(path: str, crop_cfgs: list[dict], center: bool) -> None:
    divisor = 2 if center else 3
    try:
        with Image.open(path) as img:
            w, h = img.size
            stem, parent = Path(path).stem, Path(path).parent
            for cfg in crop_cfgs:
                aw, ah = cfg["aspect"]
                target = aw / ah
                if (w / h) > target:
                    nw = int(h * target); left = (w - nw) // divisor
                    box = (left, 0, left + nw, h)
                else:
                    nh = int(w / target); top = (h - nh) // divisor
                    box = (0, top, w, top + nh)
                cpath = str(parent / f"{stem}{cfg['suffix']}.jpg")
                img.crop(box).save(cpath, "JPEG", quality=95)
                _copy_exif_no_dims(path, cpath)
    except Exception as e:
        logging.error("Crop ব্যর্থ [%s]: %s", path, e)


def move_all(records, folders, quarantine_dir, config, dry_run: bool) -> None:
    """
    🔴 FIXED: crop এখন KEEPER category-দের জন্য (duplicate নয়) এবং opt-in।
    🟠 FIXED: category None হলে silent Reject নয় → "Review" + WARNING।
    """
    crop_cats = set(config.get("crop_categories", []))
    do_crop = config.get("generate_crops", False)
    logger = logging.getLogger("funnel")

    for rec in tqdm(records, desc="Moving"):
        if rec.category is None:
            logger.warning("category None [%s] → Review (silent-loss এড়াতে)", rec.file)
            rec.category = "Review"

        cat = rec.category
        if dry_run:
            rec.final_path = os.path.join(folders[cat], rec.file)
            continue

        moved = _safe_move(rec.path, os.path.join(folders[cat], rec.file), quarantine_dir)
        rec.final_path = moved
        if moved and do_crop and cat in crop_cats:
            _generate_crops(moved, config["crop_versions"], config.get("crop_center", True))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৮. LAYER E — Reporter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def report(records, out_dir: Path, dry_run: bool) -> str:
    csv_path = out_dir / f"audit_v6_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["File", "Source", "Category", "GroupID", "IsBestOfGroup",
                      "Sharpness", "Brightness", "Composition", "NoiseSigma",
                      "AIScore", "RejectReason", "FinalPath"])
        for r in records:
            wri.writerow([r.file, r.source, r.category or "Unknown", r.group_id,
                          r.is_best_of_group, f"{r.sharpness:.2f}", f"{r.brightness:.2f}",
                          f"{r.composition:.2f}", f"{r.noise_sigma:.2f}",
                          f"{r.ai_score:.2f}", r.reject_reason or "", r.final_path or ""])

    counts = Counter(r.category or "Unknown" for r in records)
    log = logging.getLogger("funnel")
    log.info("\n" + "=" * 55)
    log.info("✅ মাস্টার ফানেল v6.0 সম্পন্ন%s", " (ড্রাই-রান)" if dry_run else "")
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        log.info("  %-14s  %d", cat, cnt)
    log.info("  CSV: %s", csv_path)
    log.info("=" * 55)
    return str(csv_path)


def report_calibration(records, out_dir: Path) -> str:
    """--calibrate: metric distribution ছাপে, threshold ঠিক করতে। কিছু move করে না।"""
    log = logging.getLogger("funnel")
    csv_path = out_dir / f"calibrate_v6_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wri = csv.writer(f)
        wri.writerow(["File", "Source", "Sharpness", "Brightness", "NoiseSigma", "Readable"])
        for r in records:
            wri.writerow([r.file, r.source, f"{r.sharpness:.2f}", f"{r.brightness:.2f}",
                          f"{r.noise_sigma:.2f}", r.category != "Reject"])

    def pct(vals):
        a = np.array(vals)
        return {p: round(float(np.percentile(a, p)), 2)
                for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)} if len(a) else {}

    ok = [r for r in records if r.category != "Reject"]
    log.info("\n" + "=" * 55)
    log.info("📊 CALIBRATION — readable %d / total %d", len(ok), len(records))
    for name, key in (("Sharpness", "sharpness"), ("Brightness", "brightness"),
                      ("NoiseSigma", "noise_sigma")):
        log.info("  %s percentiles: %s", name, pct([getattr(r, key) for r in ok]))
    cfg = DEFAULT_CONFIG
    log.info("  বর্তমান gate: sharp≥%.1f  bright∈[%.0f,%.0f]  noise≤%.1f",
             cfg["min_sharpness"], cfg["min_brightness"], cfg["max_brightness"],
             cfg["max_noise_sigma"])
    log.info("  → নিচের percentile দেখে threshold ঠিক করুন (ভালো ছবি কাটছে কিনা)।")
    log.info("  CSV: %s", csv_path)
    log.info("=" * 55)
    return str(csv_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ৯. অর্কেস্ট্রেটর
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sources = [Path(s) for s in args.source]
    out_dir = Path(args.output) if args.output else sources[0]
    logger = setup_logging(out_dir)
    mode = "CALIBRATE" if args.calibrate else ("DRY-RUN" if args.dry_run else "LIVE")
    logger.info("🚀 মাস্টার ফানেল v6.0 [%s] — sources: %s", mode, [str(s) for s in sources])

    image_pairs = collect_images(sources, logger)
    if not image_pairs:
        logger.error("কোনো ছবি পাওয়া যায়নি — শেষ।"); sys.exit(1)
    logger.info("মোট ছবি: %d", len(image_pairs))

    # ── CALIBRATE: model/move ছাড়া শুধু metric ──
    if args.calibrate:
        recs = extract_all(image_pairs, None, None, None, config, score_ai=False)
        report_calibration(recs, out_dir)
        return

    # ── ফোল্ডার প্রস্তুতি (live only) ──
    folders = {cat: os.path.join(str(out_dir), cat) for cat in config["categories"]}
    quarantine_dir = os.path.join(str(out_dir), "Quarantine")
    if not args.dry_run:
        for d in list(folders.values()) + [quarantine_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # ── মডেল ──
    device = detect_device()
    processor, model = load_ai_model(args.model, device)

    # ── A: Extract ──
    logger.info("LAYER A: Extraction (batch=%d)…", config["batch_size"])
    records = extract_all(image_pairs, processor, model, device, config, score_ai=True)

    # ── B: Deduplicate ──
    logger.info("LAYER B: Dedup (threshold=%d, transitive)…", config["hash_distance_threshold"])
    keepers, dups = deduplicate(records, config["hash_distance_threshold"])
    logger.info("Keepers: %d | Duplicates: %d", len(keepers), len(dups))

    # ── C: Classify ──
    logger.info("LAYER C: Weighted K-Means (n=%d)…", min(len(RANK_LABELS), len(keepers)))
    classify(keepers)

    # ── D: Move ──
    logger.info("LAYER D: File ops%s…", " (DRY RUN)" if args.dry_run else "")
    move_all(records, folders, quarantine_dir, config, args.dry_run)

    # ── E: Report ──
    logger.info("LAYER E: Audit CSV…")
    report(records, out_dir, args.dry_run)


if __name__ == "__main__":
    main()

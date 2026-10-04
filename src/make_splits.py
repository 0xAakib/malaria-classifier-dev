"""
F01-I2, milestone 2 — frozen splits, BBBC041 per-cell crops, canonical labels.

Produces the three things every later milestone reads, and nothing else:

  1. One frozen, stratified 80/20 train/test split per dataset, split by
     SOURCE IMAGE, never by cell — cells cut from the same smear never straddle
     the split.
  2. BBBC041 per-cell crops, cut from the full-field smears using the per-cell
     bounding boxes.
  3. One canonical label mapping (0 = uninfected, 1 = infected), and a table
     translating every source's own labels into it — both datasets and both
     found checkpoints, which disagree with each other.

Label-fraction subsets (100/25/10/2%) are milestone 3. They are drawn from the
`train` rows of the split files written here and must never touch `test`.

Run on Colab or locally, from the repo root:

    pip install pillow numpy pandas scikit-learn
    python src/make_splits.py --out data

Outputs, under --out:

    nih/cell_images/...            the NIH crops, extracted
    bbbc041/crops/*.png            one PNG per annotated cell
    splits/nih.csv                 path, label, source_image, patient, split
    splits/bbbc041.csv             path, label, category, source_image, bbox, split
    split_counts.csv               dataset, split, label, count — one row per cell of the 2x2x2
    label_map.json                 the canonical mapping and every translation into it
    split_report.json              counts per split, and a fingerprint of each test set

The fingerprint is what makes the split "frozen": re-running with the same
--seed must reproduce the same hash, and later milestones should check it before
evaluating anything.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

NIH_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
BBBC041_URL = "https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip"

# ── 3. The canonical mapping ─────────────────────────────────────────────────
# Infected is the positive class, so sklearn's defaults (pos_label=1) score the
# thing the study is about without anyone having to remember to pass it.
CANONICAL = {0: "uninfected", 1: "infected"}

BBBC041_CATEGORY = {
    "gametocyte": 1,
    "ring": 1,
    "trophozoite": 1,
    "schizont": 1,
    "red blood cell": 0,
    "leukocyte": 0,
    # Annotators could not decide. Excluded rather than guessed: a guess would put
    # label noise into the minority class, where there are ~2.4k cells in total.
    "difficult": None,
}

LABEL_MAP = {
    "canonical": CANONICAL,
    "translations": {
        "nih_folder": {"Parasitized": 1, "Uninfected": 0},
        # The HF mirror ships `label` as a bare int64 with no class names. Its
        # polarity was established by looking, not assumed: label-0 cells carry a
        # dark stained body inside the cell far more often than label-1 cells
        # (interior 1st-percentile/median green ratio 0.51 vs 0.95; P(label-0
        # darker) = 0.987 on the first 200 rows). So 0 = parasitized.
        "hf_dpdl_benchmark_malaria": {"0": 1, "1": 0},
        "bbbc041_category": {k: v for k, v in BBBC041_CATEGORY.items()},
        # Read from each checkpoint's config.json. THE TWO DISAGREE: Sadou puts
        # malaria at 1, PlasmoVision puts it at 0. Loading either without this
        # table silently inverts one of the four methods.
        "Sadou/malaria-detector-dinov2-tanzania": {
            "id2label": {"0": "sain", "1": "paludisme"},
            "to_canonical": {"0": 0, "1": 1},
        },
        "ngohjuniormbah/plasmovision-malaria-ai": {
            "id2label": {"0": "Parasitized", "1": "Uninfected"},
            "to_canonical": {"0": 1, "1": 0},
        },
        # ResNet-18 (ImageNet) and the from-scratch CNN get new two-way heads, so
        # they are built on CANONICAL directly and need no translation.
    },
}


def download(url: str, dest: Path, stalls: int = 12) -> Path:
    """
    Resumable, and size-checked before the file gets its real name.

    The Broad server closes long transfers mid-stream without an error — the read
    just ends — so "the loop finished" does not mean "the file is complete".
    Each attempt resumes from the bytes already on disk with a Range request.

    It also goes through spells of answering with nothing at all, so a retry
    straight away burns attempts for no progress. Only attempts that add no bytes
    count against `stalls`, and each one waits longer than the last.
    """
    if dest.exists():
        print(f"  cached  {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        total = int(r.headers["content-length"])
    print(f"  fetch   {url}  ({total / 1e9:.2f} GB)")

    stalled = 0
    while stalled < stalls:
        done = tmp.stat().st_size if tmp.exists() else 0
        if done >= total:
            break
        before = done
        req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab") as f:
                if done and r.status != 206:
                    sys.exit(f"✗ {url} ignored the Range request; cannot resume")
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"\r          {done / total:6.1%}", end="", flush=True)
        except (OSError, TimeoutError) as e:
            print(f"\n          dropped at {done / total:.1%} ({e.__class__.__name__})")
        if done >= total:
            break
        stalled = 0 if done > before else stalled + 1
        wait = min(300, 5 * 2**stalled)
        print(f"\n          resuming in {wait}s ({stalled}/{stalls} stalls)")
        time.sleep(wait)
    print()

    got = tmp.stat().st_size
    if got != total:
        sys.exit(f"✗ {url}: have {got} of {total} bytes after {stalls} stalled attempts; re-run to resume")
    tmp.rename(dest)  # only a complete download ever gets the real name
    return dest


# ── NIH ──────────────────────────────────────────────────────────────────────
def build_nih(out: Path, cache: Path) -> pd.DataFrame:
    """
    The original NIH release, not the HF mirror the brief names.

    They are the same 27,558 crops, but the mirror dropped the filenames — and
    the filename is the only place the source image is recorded
    (`C100P61ThinF_IMG_20150918_144104_cell_162.png`: everything before `_cell_`
    is the smear it was cut from). Without it the split can only be by cell,
    which puts sibling cells from one smear on both sides of it.
    """
    zpath = download(NIH_URL, cache / "cell_images.zip")
    root = out / "nih"
    rows = []
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if not name.lower().endswith(".png"):
                continue  # the zip also carries Thumbs.db files
            folder, fname = name.split("/")[-2:]
            source = fname.split("_cell_")[0]
            rows.append(
                {
                    "path": str(Path("nih") / name),
                    "label": LABEL_MAP["translations"]["nih_folder"][folder],
                    "source_image": source,
                    # Recorded, not split on — see the report's patient_overlap.
                    "patient": source.split("_")[0].lower().split("thinf")[0],
                }
            )
        if not (root / "cell_images").exists():
            print("  extract nih")
            z.extractall(root)
    return pd.DataFrame(rows)


# ── BBBC041 ──────────────────────────────────────────────────────────────────
def build_bbbc041(out: Path, cache: Path) -> pd.DataFrame:
    """
    Pool the official training.json and test.json, then re-split by image.

    The official split is not reused: the brief fixes an 80/20 split by source
    image for both datasets, and the official one is 1,208/120 — and in different
    formats (training images are PNG, test images JPEG), so it would also be a
    compression split. `source_format` is kept per row so that stays checkable.
    """
    zpath = download(BBBC041_URL, cache / "bbbc041_malaria.zip")
    crop_dir = out / "bbbc041" / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows, excluded, degenerate = [], 0, 0
    with zipfile.ZipFile(zpath) as z:
        records = []
        for j in ("malaria/training.json", "malaria/test.json"):
            records += json.loads(z.read(j))

        for n, rec in enumerate(records, 1):
            pathname = rec["image"]["pathname"]  # "/images/<uuid>.png"
            stem = Path(pathname).stem
            img = None  # opened lazily, once per smear, only if it has a usable cell

            for k, obj in enumerate(rec["objects"]):
                label = BBBC041_CATEGORY[obj["category"]]
                if label is None:
                    excluded += 1
                    continue
                if img is None:
                    img = Image.open(io.BytesIO(z.read("malaria" + pathname))).convert("RGB")
                bb = obj["bounding_box"]
                # Boxes are (row, col); PIL wants (left, top, right, bottom).
                left, top = max(0, bb["minimum"]["c"]), max(0, bb["minimum"]["r"])
                right, bottom = min(img.width, bb["maximum"]["c"]), min(img.height, bb["maximum"]["r"])
                if right - left < 8 or bottom - top < 8:
                    degenerate += 1
                    continue

                rel = Path("bbbc041") / "crops" / f"{stem}_{k:04d}.png"
                if not (out / rel).exists():
                    img.crop((left, top, right, bottom)).save(out / rel)
                rows.append(
                    {
                        "path": str(rel),
                        "label": label,
                        "category": obj["category"],
                        "source_image": stem,
                        "source_format": Path(pathname).suffix.lstrip("."),
                        "bbox_ltrb": f"{left},{top},{right},{bottom}",
                    }
                )
            if n % 100 == 0 or n == len(records):
                print(f"\r  crop    {n}/{len(records)} smears, {len(rows)} cells", end="", flush=True)
    print()
    print(f"          excluded {excluded} 'difficult' cells, {degenerate} boxes under 8px")
    return pd.DataFrame(rows)


# ── 1. The split ─────────────────────────────────────────────────────────────
def split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    80/20 by source image, stratified on the cell label.

    StratifiedGroupKFold with five folds, keeping fold 0 as test: groups keep
    every cell of a smear on one side, and stratification keeps the infected rate
    of test close to the whole dataset's — which matters on BBBC041, where
    infected cells are under 3% and a careless image split can leave test with
    almost none.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], groups=df["source_image"]))
    df = df.copy()
    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"
    return df


def report(name: str, df: pd.DataFrame) -> dict:
    train, test = df[df.split == "train"], df[df.split == "test"]
    leaked = set(train.source_image) & set(test.source_image)
    # A hard stop, not a warning: every result downstream rests on this.
    if leaked:
        sys.exit(f"✗ {name}: {len(leaked)} source images on both sides of the split")

    out = {
        "cells": len(df),
        "source_images": int(df.source_image.nunique()),
        "source_image_overlap": 0,
        "test_fingerprint": hashlib.sha256("\n".join(sorted(test.path)).encode()).hexdigest(),
    }
    for part, d in (("train", train), ("test", test)):
        out[part] = {
            "cells": len(d),
            "source_images": int(d.source_image.nunique()),
            "infected": int(d.label.sum()),
            "uninfected": int((d.label == 0).sum()),
            "infected_rate": round(float(d.label.mean()), 4),
        }
    if "patient" in df:
        # Not a failure: the brief fixes the split by source image. Reported
        # because a patient contributes several smears, so this is the leak that
        # remains, and whoever writes up the results should know its size.
        out["patient_overlap"] = len(set(train.patient) & set(test.patient))
        out["patients"] = int(df.patient.nunique())
    if "source_format" in df:
        out["test_formats"] = test.source_format.value_counts().to_dict()
    return out


def counts(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    One row per (dataset, split, label), always all eight, in a fixed order.

    Counted from the split frames rather than copied out of the report, so the
    table and the split files cannot disagree. A combination with no cells still
    gets a row with count 0 — a missing row would hide exactly the failure (no
    infected cells in a test split) this table exists to show.
    """
    rows = []
    for name, df in frames.items():
        for part in ("train", "test"):
            d = df[df.split == part]
            for code in (1, 0):
                rows.append(
                    {"dataset": name, "split": part, "label": CANONICAL[code], "count": int((d.label == code).sum())}
                )
    return pd.DataFrame(rows, columns=["dataset", "split", "label", "count"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=None, help="where the zips go (default: <out>/_downloads)")
    ap.add_argument("--seed", type=int, default=0, help="split seed; change it and the split is no longer the frozen one")
    args = ap.parse_args()

    out: Path = args.out
    cache: Path = args.cache or out / "_downloads"
    (out / "splits").mkdir(parents=True, exist_ok=True)

    print("NIH")
    nih = split(build_nih(out, cache), args.seed)
    print("BBBC041")
    bbbc = split(build_bbbc041(out, cache), args.seed)

    nih.to_csv(out / "splits" / "nih.csv", index=False)
    bbbc.to_csv(out / "splits" / "bbbc041.csv", index=False)
    (out / "label_map.json").write_text(json.dumps(LABEL_MAP, indent=2))

    rep = {"seed": args.seed, "nih": report("nih", nih), "bbbc041": report("bbbc041", bbbc)}
    (out / "split_report.json").write_text(json.dumps(rep, indent=2))
    counts({"nih": nih, "bbbc041": bbbc}).to_csv(out / "split_counts.csv", index=False)

    for name in ("nih", "bbbc041"):
        r = rep[name]
        print(
            f"\n{name}: {r['cells']} cells from {r['source_images']} source images"
            f"\n  train {r['train']['cells']:>6} cells  infected {r['train']['infected_rate']:.2%}"
            f"\n  test  {r['test']['cells']:>6} cells  infected {r['test']['infected_rate']:.2%}"
            f"\n  source-image overlap 0 · test fingerprint {r['test_fingerprint'][:16]}"
        )
        if "patient_overlap" in r:
            print(f"  patients on both sides: {r['patient_overlap']} of {r['patients']}")
    print(
        f"\n✓ wrote {out / 'splits'}, {out / 'split_counts.csv'}, {out / 'label_map.json'}, {out / 'split_report.json'}"
    )

    if args.cache is None:
        print(f"  (the zips are still in {cache} — delete it to reclaim ~2.6 GB)")


if __name__ == "__main__":
    main()

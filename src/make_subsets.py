"""
F01-I2, milestone 3 — stratified label-fraction subsets of the frozen train split.

Reads the split files milestone 2 wrote and, for every dataset and seed, draws
the 100/25/10/2% subsets that milestone 4 fine-tunes on.

  1. Train rows only. The test split is never read into the sampler; the script
     checks the test set's fingerprint against split_report.json first, so it
     refuses to run against a split that is no longer the frozen one.
  2. Stratified: each class is sampled separately at the same fraction, so every
     subset keeps the train split's infected rate. On BBBC041 that rate is under
     3%, and a plain random 2% draw could leave a seed with almost no infected
     cells.
  3. Nested: within a seed, the 2% subset is inside the 10%, which is inside the
     25%. Moving along the curve only adds examples, so a change between two
     points is the extra labels, not a different draw.

Run from the repo root:

    pip install numpy pandas
    python src/make_subsets.py --data data

Outputs, under --data:

    subsets/nih.csv          one row per train cell: path, label, source_image,
    subsets/bbbc041.csv      and per seed the smallest fraction it belongs to
    subset_counts.csv        dataset, fraction, seed, label, count — 32 rows

A cell is in the (fraction f, seed s) subset when its `min_fraction_s<s>` is <= f.
Use `load_subset` below rather than repeating that rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ("nih", "bbbc041")
FRACTIONS = (1.0, 0.25, 0.1, 0.02)  # largest first: each subset is cut from the one before
SEEDS = (0, 1)
LABELS = {1: "infected", 0: "uninfected"}  # canonical mapping from label_map.json


def check_frozen(data: Path, name: str, split_df: pd.DataFrame) -> None:
    """Stop unless the test set is the one milestone 2 fingerprinted."""
    expected = json.loads((data / "split_report.json").read_text())[name]["test_fingerprint"]
    test_paths = sorted(split_df.loc[split_df.split == "test", "path"])
    got = hashlib.sha256("\n".join(test_paths).encode()).hexdigest()
    if got != expected:
        sys.exit(f"✗ {name}: test fingerprint {got[:16]} does not match split_report.json ({expected[:16]})")


def sample_size(n: int, fraction: float) -> int:
    # At least one cell per class, so no subset silently drops a class.
    return n if fraction == 1.0 else max(1, round(n * fraction))


def assign(train: pd.DataFrame, seed: int) -> pd.Series:
    """
    For one seed, the smallest fraction each train cell belongs to.

    Each class is shuffled once; the first k cells of that order form the
    subset at the fraction whose size is k. Smaller fractions take a prefix of
    the same order, which is what makes the subsets nested.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(np.nan, index=train.index)
    for code in LABELS:
        idx = train.index[train.label == code].to_numpy()
        order = idx[rng.permutation(len(idx))]
        for fraction in FRACTIONS:
            out[order[: sample_size(len(idx), fraction)]] = fraction
    return out


def load_subset(data: Path, name: str, fraction: float, seed: int) -> pd.DataFrame:
    """The cells of one (dataset, fraction, seed) subset. Milestone 4 reads through this."""
    df = pd.read_csv(data / "subsets" / f"{name}.csv")
    return df[df[f"min_fraction_s{seed}"] <= fraction]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    data: Path = args.data
    (data / "subsets").mkdir(exist_ok=True)

    rows = []
    for name in DATASETS:
        split_df = pd.read_csv(data / "splits" / f"{name}.csv")
        check_frozen(data, name, split_df)

        # Sorted so the draw depends only on the seed, not on the CSV's row order.
        train = (
            split_df[split_df.split == "train"][["path", "label", "source_image"]]
            .sort_values("path")
            .reset_index(drop=True)
        )
        for seed in SEEDS:
            train[f"min_fraction_s{seed}"] = assign(train, seed)
        train.to_csv(data / "subsets" / f"{name}.csv", index=False)

        full_rate = train.label.mean()
        print(f"{name}: {len(train)} train cells, infected {full_rate:.2%}")
        for fraction in FRACTIONS:
            for seed in SEEDS:
                subset = train[train[f"min_fraction_s{seed}"] <= fraction]
                for code, label in LABELS.items():
                    rows.append(
                        {
                            "dataset": name,
                            "fraction": str(fraction),
                            "seed": seed,
                            "label": label,
                            "count": int((subset.label == code).sum()),
                        }
                    )
                print(
                    f"  {fraction:>5}  seed {seed}  {len(subset):>6} cells"
                    f"  infected {int(subset.label.sum()):>5} ({subset.label.mean():.2%})"
                )

    counts = pd.DataFrame(rows, columns=["dataset", "fraction", "seed", "label", "count"])
    counts.to_csv(data / "subset_counts.csv", index=False)
    print(f"\n✓ wrote {data / 'subsets'}, {data / 'subset_counts.csv'} ({len(counts)} rows)")


if __name__ == "__main__":
    main()

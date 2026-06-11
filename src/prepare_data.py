"""Task 2 — Data Preparation & Normalisation.

A reusable data-prep script for binary sentiment classification.

It will:
  1. Load a publicly available dataset (SST-2 by default; IMDB or a local CSV also work).
  2. Inspect the raw data (size, structure, class distribution, quality issues).
  3. Clean / normalise the text (whitespace, case, duplicates, missing values, length).
  4. Encode labels and save an ``id2label.json`` mapping file.
  5. Save a stratified train/val/test split locally (kept out of git via .gitignore).

Only ``id2label.json`` is meant to be committed — the prepared CSVs stay local.

Usage
-----
    python src/prepare_data.py --dataset sst2 --sample-size 20000 --out-dir data
    python src/prepare_data.py --dataset imdb --out-dir data
    python src/prepare_data.py --dataset path/to/my.csv --text-col review --label-col sentiment
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

# datasets / sklearn are imported lazily inside functions so that `--help`
# and CSV-only runs work even in a minimal environment.

WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(dataset: str, text_col: str, label_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Return a raw (text, label) DataFrame plus the ordered label names.

    ``dataset`` may be ``sst2``, ``imdb`` or a path to a local CSV file.
    """
    if dataset.lower() == "sst2":
        from datasets import load_dataset

        # SST-2's public test split has hidden labels (-1), so we build our own
        # splits from the labelled train split downstream.
        ds = load_dataset("stanfordnlp/sst2", split="train")
        label_names = ds.features["label"].names  # ['negative', 'positive']
        df = pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})
        return df, list(label_names)

    if dataset.lower() == "imdb":
        from datasets import load_dataset

        ds = load_dataset("imdb", split="train")
        label_names = ds.features["label"].names  # ['neg', 'pos']
        df = pd.DataFrame({"text": ds["text"], "label": ds["label"]})
        return df, ["negative", "positive"] if label_names == ["neg", "pos"] else list(label_names)

    # Otherwise treat ``dataset`` as a local CSV path.
    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"Dataset '{dataset}' is not 'sst2'/'imdb' nor an existing CSV path.")
    df = pd.read_csv(dataset)
    df = df.rename(columns={text_col: "text", label_col: "label"})
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")
    # Derive label names from the non-missing labels (integer-like -> clean "0"/"1").
    uniq = sorted(pd.Series(df["label"]).dropna().unique().tolist())
    label_names = [
        str(int(u)) if isinstance(u, (int, float)) and float(u).is_integer() else str(u)
        for u in uniq
    ]
    return df[["text", "label"]], label_names


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def inspect(df: pd.DataFrame) -> Dict:
    """Collect basic dataset statistics and quality signals."""
    text_lengths = df["text"].astype(str).str.len()
    # Use string keys + dropna so raw data with missing/non-integer labels never crashes.
    class_counts = df["label"].value_counts(dropna=True)
    stats = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "n_missing_text": int(df["text"].isna().sum()),
        "n_missing_label": int(df["label"].isna().sum()),
        "n_empty_text": int((df["text"].astype(str).str.strip() == "").sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "text_len_min": int(text_lengths.min()),
        "text_len_mean": round(float(text_lengths.mean()), 2),
        "text_len_max": int(text_lengths.max()),
    }
    return stats


# --------------------------------------------------------------------------- #
# Cleaning / normalisation
# --------------------------------------------------------------------------- #
def normalise_text(text: str, lowercase: bool) -> str:
    """Strip HTML breaks, collapse whitespace, optionally lowercase."""
    text = str(text)
    text = text.replace("<br />", " ").replace("<br>", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def clean(df: pd.DataFrame, lowercase: bool, min_len: int, max_len: int) -> pd.DataFrame:
    """Apply cleaning steps and return a deduplicated, valid DataFrame."""
    df = df.copy()
    # 1. Drop rows with missing text or label.
    df = df.dropna(subset=["text", "label"])
    # 2. Coerce labels to int.
    df["label"] = df["label"].astype(int)
    # 3. Normalise text.
    df["text"] = df["text"].map(lambda t: normalise_text(t, lowercase))
    # 4. Remove empty / too-short / too-long after normalisation.
    lengths = df["text"].str.len()
    df = df[(lengths >= min_len) & (lengths <= max_len)]
    # 5. Drop exact duplicate (text, label) pairs.
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Stratified down-sampling that preserves class balance."""
    if sample_size <= 0 or sample_size >= len(df):
        return df
    frac = sample_size / len(df)
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )


def split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 80/10/10 train/val/test split."""
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a sentiment dataset for fine-tuning.")
    p.add_argument("--dataset", default="sst2", help="'sst2', 'imdb', or path to a CSV.")
    p.add_argument("--text-col", default="text", help="Text column name (CSV only).")
    p.add_argument("--label-col", default="label", help="Label column name (CSV only).")
    p.add_argument("--sample-size", type=int, default=20000, help="Stratified cap on rows (0 = keep all).")
    p.add_argument("--min-len", type=int, default=1, help="Min text length (chars) after cleaning.")
    p.add_argument("--max-len", type=int, default=1000, help="Max text length (chars) after cleaning.")
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing.")
    p.add_argument("--out-dir", default="data", help="Where to save prepared CSVs + stats.")
    p.add_argument("--mapping-path", default="id2label.json", help="Where to save id2label.json.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/5] Loading dataset: {args.dataset}")
    raw_df, label_names = load_raw(args.dataset, args.text_col, args.label_col)

    print("[2/5] Inspecting raw data")
    raw_stats = inspect(raw_df)
    print(json.dumps(raw_stats, indent=2))

    print("[3/5] Cleaning & normalising")
    clean_df = clean(raw_df, lowercase=not args.no_lowercase, min_len=args.min_len, max_len=args.max_len)
    clean_df = stratified_sample(clean_df, args.sample_size, args.seed)
    clean_stats = inspect(clean_df)
    print(f"    rows: {raw_stats['n_rows']} -> {clean_stats['n_rows']} "
          f"(removed {raw_stats['n_rows'] - clean_stats['n_rows']})")

    print("[4/5] Encoding labels & writing id2label.json")
    id2label = {str(i): name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    with open(args.mapping_path, "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
    print(f"    id2label = {id2label}")

    print("[5/5] Splitting & saving (stratified 80/10/10)")
    train_df, val_df, test_df = split(clean_df, args.seed)
    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)

    summary = {
        "dataset": args.dataset,
        "label_names": label_names,
        "id2label": id2label,
        "label2id": label2id,
        "raw_stats": raw_stats,
        "clean_stats": clean_stats,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "cleaning": {
            "lowercase": not args.no_lowercase,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "deduplicated": True,
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"Saved CSVs + stats.json -> {args.out_dir}/  |  mapping -> {args.mapping_path}")


if __name__ == "__main__":
    main()

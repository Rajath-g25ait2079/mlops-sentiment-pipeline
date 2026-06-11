"""Tasks 3-5 — Fine-tune DistilBERT, track with W&B, push to the Hugging Face Hub.

Designed to run inside a Kaggle GPU notebook (reads tokens from Kaggle Secrets) or
locally (reads tokens from environment variables). Two experiment versions are
produced by changing a single hyperparameter on the command line, e.g. the
learning rate:

    # Version 1
    python src/train.py --version v1 --learning-rate 3e-5 --epochs 3 --batch-size 16 \
        --push-to-hub --hf-repo Rajath-g25ait2079/distilbert-sst2-sentiment

    # Version 2 (only the learning rate changes)
    python src/train.py --version v2 --learning-rate 1e-5 --epochs 3 --batch-size 16

Logged to W&B for every run: training loss, validation loss, accuracy, weighted F1,
and all hyperparameters. The Hugging Face model URL is written to the run summary.

Tokens are NEVER hardcoded — they come from Kaggle Secrets or environment variables.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np


# --------------------------------------------------------------------------- #
# Secrets (Kaggle Secrets first, then environment variables) — Task 4
# --------------------------------------------------------------------------- #
def load_secrets() -> None:
    """Populate WANDB_API_KEY / HF_TOKEN from Kaggle Secrets if available."""
    try:
        from kaggle_secrets import UserSecretsClient  # only present on Kaggle

        secrets = UserSecretsClient()
        for key in ("WANDB_API_KEY", "HF_TOKEN"):
            try:
                os.environ.setdefault(key, secrets.get_secret(key))
            except Exception:  # noqa: BLE001 - secret may not be set
                pass
    except Exception:  # noqa: BLE001 - not running on Kaggle
        pass


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_datasets(data_dir: str, mapping_path: str):
    """Load prepared CSV splits (running prepare_data first if they are absent)."""
    import pandas as pd
    from datasets import Dataset, DatasetDict

    needed = [os.path.join(data_dir, f) for f in ("train.csv", "val.csv", "test.csv")]
    if not all(os.path.isfile(p) for p in needed):
        print("Prepared CSVs not found — generating them with prepare_data.py ...")
        import prepare_data  # same src/ directory

        raw_df, label_names = prepare_data.load_raw("sst2", "text", "label")
        clean_df = prepare_data.clean(raw_df, lowercase=True, min_len=1, max_len=1000)
        clean_df = prepare_data.stratified_sample(clean_df, 20000, 42)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({str(i): n for i, n in enumerate(label_names)}, f, indent=2)
        os.makedirs(data_dir, exist_ok=True)
        tr, va, te = prepare_data.split(clean_df, 42)
        tr.to_csv(needed[0], index=False)
        va.to_csv(needed[1], index=False)
        te.to_csv(needed[2], index=False)

    dd = DatasetDict(
        train=Dataset.from_pandas(pd.read_csv(needed[0])),
        validation=Dataset.from_pandas(pd.read_csv(needed[1])),
        test=Dataset.from_pandas(pd.read_csv(needed[2])),
    )
    return dd


def load_id2label(mapping_path: str) -> Dict[int, str]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Metrics — Task 4 (accuracy + weighted F1)
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a sentiment model and track on W&B.")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--version", default="v1", help="Experiment version tag, e.g. v1 / v2.")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--mapping-path", default="id2label.json")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--wandb-project", default="mlops-assignment3")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hf-repo", default="Rajath-g25ait2079/distilbert-sst2-sentiment")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_secrets()

    import wandb
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    id2label = load_id2label(args.mapping_path)
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    # ----- Task 3: tokenizer + model with the correct number of labels -----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ----- Data -----
    dd = build_datasets(args.data_dir, args.mapping_path)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    dd = dd.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ----- Task 4: initialise a W&B run with all hyperparameters in the config -----
    if os.environ.get("WANDB_API_KEY"):
        wandb.login()
    run_name = f"run-{args.version}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "version": args.version,
            "platform": "Kaggle",
            "dataset": "SST-2",
        },
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb",
        run_name=run_name,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dd["train"],
        eval_dataset=dd["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final held-out test metrics (logged to W&B as well).
    test_metrics = trainer.evaluate(dd["test"], metric_key_prefix="test")
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    wandb.log(test_metrics)

    # ----- Task 5: push best model + tokenizer to the Hub and log the URL -----
    if args.push_to_hub and os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])
        model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        hf_url = f"https://huggingface.co/{args.hf_repo}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"Pushed model to {hf_url}")
    else:
        print("Skipping push_to_hub (flag off or HF_TOKEN missing).")

    wandb.finish()


if __name__ == "__main__":
    main()

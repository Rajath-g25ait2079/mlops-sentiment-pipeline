"""Tasks 6-7 — Inference entry point used by both Docker and GitHub Actions.

Loads the published model from the Hugging Face Hub and classifies a piece of text.
The model repo is configurable so the same code works for any fine-tuned checkpoint:

  * ``HF_MODEL_NAME`` env var (set from the Docker build ARG) — which model to load.
  * ``INPUT_TEXT``  env var or first CLI argument — the text to classify.
  * ``HF_TOKEN``    env var — only needed if the model repo is private.

Usage
-----
    INPUT_TEXT="This film was wonderful!" python src/inference.py
    python src/inference.py "I would not recommend it."
"""
from __future__ import annotations

import json
import os
import sys

DEFAULT_MODEL = "Rajath-g25ait2079/distilbert-sst2-sentiment"
DEFAULT_TEXT = "This movie was an absolute masterpiece, I loved every minute."


def get_input_text() -> str:
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return " ".join(sys.argv[1:])
    return os.environ.get("INPUT_TEXT", DEFAULT_TEXT)


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = os.environ.get("HF_MODEL_NAME", DEFAULT_MODEL)
    token = os.environ.get("HF_TOKEN")  # optional; only for private repos
    text = get_input_text()

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())

    # Prefer the labels stored on the model config; fall back to the index.
    id2label = getattr(model.config, "id2label", None) or {pred_id: str(pred_id)}
    label = id2label.get(pred_id, id2label.get(str(pred_id), str(pred_id)))

    result = {
        "input": text,
        "label": label,
        "score": round(float(probs[pred_id].item()), 4),
        "all_scores": {
            id2label.get(i, id2label.get(str(i), str(i))): round(float(p), 4)
            for i, p in enumerate(probs.tolist())
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

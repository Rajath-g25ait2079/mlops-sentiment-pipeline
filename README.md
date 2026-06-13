# End-to-End MLOps Pipeline — Sentiment Analysis (DistilBERT + SST-2)

> **MLOps | PGD AI Program | IIT Jodhpur — Group 21**
> Docker · GitHub Actions · Kaggle · Weights & Biases · Hugging Face

A complete, production-style MLOps pipeline that fine-tunes a compact pre-trained
transformer (`distilbert-base-uncased`) for binary sentiment classification on the
**SST-2** dataset, containerises inference with Docker, runs versioned experiments
on Kaggle, automates CI + inference with GitHub Actions, and tracks every run on
Weights & Biases.

The point of this project is **command of the full MLOps workflow**, not peak
accuracy. The model is treated as a black box.

---

## Live Links

| Component | Link |
|---|---|
| GitHub Repository | https://github.com/Rajath-g25ait2079/mlops-sentiment-pipeline |
| Kaggle Notebook — Version 1 | https://www.kaggle.com/rajathsmg25ait2079/mlops-sentiment-v1 |
| Kaggle Notebook — Version 2 | https://www.kaggle.com/rajathsmg25ait2079/mlops-sentiment-v2 |
| Hugging Face Model | https://huggingface.co/Rajath-g25ait2079/distilbert-sst2-sentiment |
| Docker Image (Docker Hub) | https://hub.docker.com/r/rajathsmg25ait2079/mlops-a3-inference |
| W&B Project Dashboard | https://wandb.ai/g25ait2079/mlops-assignment3 |

> ⚠️ Replace placeholder notebook slugs with the real ones once published, and make
> every link **public** before submission. Broken or private links score zero.

---

## Repository Structure

```
mlops-sentiment-pipeline/
├── README.md                  # this file
├── LICENSE                    # MIT
├── .gitignore                 # ignores data/ and secrets; keeps id2label.json
├── .dockerignore
├── requirements.txt           # slim: inference + CI (torch, transformers)
├── requirements-train.txt     # full training stack (datasets, wandb, sklearn…)
├── id2label.json              # label mapping (the ONLY data artifact committed)
├── Dockerfile                 # slim inference image, HF_MODEL_NAME build ARG
├── src/
│   ├── prepare_data.py        # Task 2 — clean, normalise, split, save mapping
│   ├── train.py               # Tasks 3-5 — HF Trainer + W&B + push_to_hub
│   └── inference.py           # Task 6/7 — load model from HF, classify text
├── notebooks/
│   ├── kaggle_train_v1.ipynb  # experiment v1 (lr=3e-5)
│   └── kaggle_train_v2.ipynb  # experiment v2 (lr=1e-5)  ← one hyperparam changed
├── .github/workflows/
│   ├── ci.yml                 # lint on push→develop / PR→main
│   └── inference.yml          # manual inference (workflow_dispatch)
├── RUNBOOK.md                 # exact steps to execute the external platforms
└── SUBMISSION_CHECKLIST.md    # marks-mapped checklist
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/Rajath-g25ait2079/mlops-sentiment-pipeline.git
cd mlops-sentiment-pipeline

# 2. (Optional) virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3a. Inference only (small)
pip install -r requirements.txt

# 3b. Full training / data-prep stack
pip install -r requirements-train.txt
```

### How to run each script

**Data preparation (Task 2)** — downloads SST-2, cleans/normalises it, writes the
label mapping and the prepared splits to `data/` (gitignored):

```bash
python src/prepare_data.py --dataset sst2 --sample-size 20000 --out-dir data
# produces: id2label.json, data/train.csv, data/val.csv, data/test.csv, data/stats.json
```

**Training (Tasks 3-5)** — runs locally or, preferably, inside a Kaggle GPU notebook.
Hyperparameters are CLI flags so the two experiment versions differ by one value:

```bash
# Version 1
python src/train.py --version v1 --learning-rate 3e-5 --epochs 3 --batch-size 16 \
    --push-to-hub --hf-repo Rajath-g25ait2079/distilbert-sst2-sentiment

# Version 2 (only the learning rate changes)
python src/train.py --version v2 --learning-rate 1e-5 --epochs 3 --batch-size 16
```

> On Kaggle, secrets are read automatically (`WANDB_API_KEY`, `HF_TOKEN`); locally,
> export them as environment variables first. **Never hardcode tokens.**

**Inference (Tasks 6-7)** — pulls the published model from Hugging Face and classifies text:

```bash
INPUT_TEXT="This movie was an absolute masterpiece." python src/inference.py
# → {"label": "positive", "score": 0.99, ...}
```

### Run inference with Docker

```bash
# Build (model name is a build ARG with a sensible default)
docker build --build-arg HF_MODEL_NAME=Rajath-g25ait2079/distilbert-sst2-sentiment \
             -t rajathsmg25ait2079/mlops-a3-inference:latest .

# Run
docker run --rm -e INPUT_TEXT="I did not enjoy this at all." \
           rajathsmg25ait2079/mlops-a3-inference:latest

# Or pull the published image straight from Docker Hub
docker run --rm -e INPUT_TEXT="Loved it!" \
           rajathsmg25ait2079/mlops-a3-inference:latest
```

---

## CI / CD (GitHub Actions)

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | push to `develop`, PR to `main` | Install deps + `flake8` lint of `src/` |
| `inference.yml` | manual (`workflow_dispatch`) | Pull HF model, classify the text you enter |

Training is **never** run in GitHub Actions — that happens only on Kaggle GPUs.

---

## Experiment Versions

| | Version 1 | Version 2 |
|---|---|---|
| Base model | `distilbert-base-uncased` | `distilbert-base-uncased` |
| Learning rate | **3e-5** | **1e-5** *(changed)* |
| Epochs | 3 | 3 |
| Batch size | 16 | 16 |
| Tracked on W&B | ✅ | ✅ |

Metrics logged per run: training loss, validation loss, accuracy, weighted F1, and
all hyperparameters. Compare side-by-side on the
[W&B dashboard](https://wandb.ai/g25ait2079/mlops-assignment3).

---

## Team — Group 21

| Roll No. | Name | Role |
|---|---|---|
| G25AIT2079 | **Rajath S M** | Lead — pipeline, training, Docker, CI/CD |
| G25AIT2069 | Nilajit Sarkar | Data preparation & W&B tracking |
| G25AIT2130 | Vishvesh Rajpurohit | Model selection & Hugging Face Hub |
| G25AIT2142 | Niketh Varma Tirumalaraju | GitHub Actions & report |

## License

Released under the [MIT License](LICENSE).

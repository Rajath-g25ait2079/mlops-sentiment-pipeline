# Task 6 — Slim inference image for the sentiment model.
# Only inference dependencies are installed; training packages are excluded.

FROM python:3.11-slim

# Which Hugging Face model to load. Overridable at build time:
#   docker build --build-arg HF_MODEL_NAME=user/model -t mlops-a3-inference .
ARG HF_MODEL_NAME=Rajath-g25ait2079/distilbert-sst2-sentiment
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

# Cleaner, smaller, more predictable runtime.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install CPU-only PyTorch first (keeps the image small), then the rest of the
# slim inference requirements. torch>=2.2 in requirements.txt is already satisfied.
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Copy only what inference needs.
COPY src/inference.py src/inference.py
COPY id2label.json .

# Optional: pre-bake the model so the container needs no network at runtime.
# Enable only after the model has been pushed to the Hub (Task 5).
# RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('${HF_MODEL_NAME}'); \
#     AutoModelForSequenceClassification.from_pretrained('${HF_MODEL_NAME}')"

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Reads INPUT_TEXT from the environment, or accepts text as a CLI argument:
#   docker run --rm -e INPUT_TEXT="Great film!" mlops-a3-inference:latest
#   docker run --rm mlops-a3-inference:latest "Terrible film."
ENTRYPOINT ["python", "src/inference.py"]

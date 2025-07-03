# ---------- Build ARGs ----------
#   TARGET=cpu  (default)  → lightweight Python base
#   TARGET=gpu            → NVIDIA CUDA base (needs nvidia-docker runtime)
# -------------------------------
ARG TARGET=cpu

# ---------- Stage 1 : base OS & Python ----------
FROM --platform=linux/amd64 python:3.11-slim AS cpu
FROM --platform=linux/amd64 nvidia/cuda:12.5.0-base-ubuntu22.04 AS gpu

# Switch to the selected base image
FROM ${TARGET} AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english

# Basic OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000
CMD ["python", "app.py"]

# Dockerfile
# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Copy and install dependencies first (Docker layer caching) ─────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────
COPY api/        ./api/
COPY models/     ./models/
COPY data/       ./data/

# ── Create reports directory ───────────────────────────────────────────────
RUN mkdir -p reports

# ── Pre-download the sentence-transformer model into the image ─────────────
# This means the container starts fast — no download on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Expose port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ───────────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

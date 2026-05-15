# api/app.py
import time
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from api.auth     import verify_api_key
from api.database import (get_analytics, get_db, get_review_queue,
                           init_db, log_prediction, mark_reviewed)
from api.llm      import generate_suggested_reply
from api.schemas  import (AnalyticsOut, BatchIn, BatchOut, BatchPredictionItem,
                           HealthOut, MetadataOut, PredictionOut, ReviewActionIn,
                           ReviewOut, SimilarIn, SimilarTicket, TicketIn)

# ── Project paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Load models at startup (once) ─────────────────────────────────────────
print("Loading models...")
try:
    from sentence_transformers import SentenceTransformer
    ENCODER     = SentenceTransformer("all-MiniLM-L6-v2")
    CAT_MODEL   = joblib.load(MODELS_DIR / "category_model.joblib")
    PRI_MODEL   = joblib.load(MODELS_DIR / "priority_model.joblib")
    SEN_MODEL   = joblib.load(MODELS_DIR / "sentiment_model.joblib")
    TFIDF_INDEX = joblib.load(MODELS_DIR / "tfidf_index.joblib")

    import pandas as pd
    CORPUS_DF   = pd.read_csv(MODELS_DIR / "training_corpus.csv")
    CORPUS_VECS = TFIDF_INDEX.transform(CORPUS_DF["text"].tolist())

    MODELS_LOADED = True
    print("All models loaded.")
except Exception as e:
    MODELS_LOADED = False
    print(f"ERROR loading models: {e}")
    raise

# ── LLM availability ───────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
LLM_ENABLED = bool(os.getenv("GROQ_API_KEY", ""))

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Support Ticket Auto-Triage",
    description=(
        "Multi-output ML API that classifies support tickets into Category, Priority, "
        "and Sentiment. Includes LLM-generated reply suggestions, similar ticket lookup, "
        "analytics dashboard data, and a low-confidence review queue."
    ),
    version="2.0.0",
)

# Allow frontend (React/Streamlit) to call this API from a different port/domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise SQLite tables on startup
init_db()


# ── Helpers ────────────────────────────────────────────────────────────────
def _build_text(subject: str, description: str) -> str:
    return f"{subject} {description}".lower().strip()


def _predict_all(text: str) -> dict:
    """Encode text and run all three classifiers. Returns raw results."""
    embedding = ENCODER.encode([text])

    cat_proba  = CAT_MODEL.predict_proba(embedding)[0]
    cat_idx    = int(np.argmax(cat_proba))
    category   = CAT_MODEL.classes_[cat_idx]
    confidence = float(cat_proba[cat_idx])

    priority  = str(PRI_MODEL.predict(embedding)[0])
    sentiment = str(SEN_MODEL.predict(embedding)[0])

    return dict(category=category, priority=priority,
                sentiment=sentiment, confidence=confidence)


def _find_similar(text: str, top_k: int = 3) -> list:
    """Cosine similarity against the training corpus using TF-IDF."""
    from sklearn.metrics.pairwise import cosine_similarity
    query_vec  = TFIDF_INDEX.transform([text])
    scores     = cosine_similarity(query_vec, CORPUS_VECS).flatten()
    top_idx    = scores.argsort()[::-1][:top_k]
    results    = []
    for i in top_idx:
        results.append(SimilarTicket(
            subject  = str(CORPUS_DF.iloc[i]["Subject"]),
            category = str(CORPUS_DF.iloc[i]["Category"]),
            score    = round(float(scores[i]), 4),
        ))
    return results


# ── Homepage ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    return HTMLResponse("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Support Auto-Triage v2</title>
  <style>
    *{box-sizing:border-box}
    body{margin:0;font-family:ui-sans-serif,system-ui,sans-serif;background:#0f172a;color:#e2e8f0}
    .wrap{max-width:860px;margin:52px auto;padding:0 20px}
    h1{font-size:1.7rem;font-weight:800;margin:0 0 6px}
    .sub{color:#94a3b8;margin:0 0 28px}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
    .card{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:18px}
    .card h3{margin:0 0 8px;font-size:1rem}
    .card p{margin:0 0 12px;color:#94a3b8;font-size:.875rem;line-height:1.5}
    a.btn{display:inline-block;padding:8px 16px;background:#3b82f6;color:#fff;
          border-radius:8px;text-decoration:none;font-size:.875rem;font-weight:500}
    code{background:#0f172a;border:1px solid #334155;padding:2px 7px;
         border-radius:5px;font-size:.8rem;color:#7dd3fc}
    .badge{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.75rem;
           font-weight:600;background:#166534;color:#bbf7d0;margin-left:8px}
    footer{margin-top:24px;color:#475569;font-size:.8rem}
  </style>
</head>
<body>
<div class="wrap">
  <h1>Customer Support Auto-Triage <span class="badge">v2.0</span></h1>
  <p class="sub">Multi-output ML · Sentence-Transformers · LLM reply suggestions · Live analytics</p>
  <div class="grid">
    <div class="card">
      <h3>API Docs</h3>
      <p>Interactive Swagger UI — try every endpoint in the browser.</p>
      <a class="btn" href="/docs">Open Swagger UI</a>
    </div>
    <div class="card">
      <h3>Endpoints</h3>
      <p>
        <code>POST /predict</code> — single ticket<br><br>
        <code>POST /predict_batch</code> — bulk CSV<br><br>
        <code>GET  /analytics</code> — dashboard data<br><br>
        <code>GET  /review</code> — low-confidence queue<br><br>
        <code>POST /similar</code> — related tickets<br><br>
        <code>GET  /health</code> · <code>GET /metadata</code>
      </p>
    </div>
    <div class="card">
      <h3>Auth</h3>
      <p>All POST routes require the header:</p>
      <code>X-API-Key: capstone-2026</code>
    </div>
  </div>
  <footer>Capstone Project · Sentence-Transformers (all-MiniLM-L6-v2) · FastAPI · SQLite</footer>
</div>
</body>
</html>
""")


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOut, tags=["System"])
def health():
    return HealthOut(
        status        = "ok",
        models_loaded = MODELS_LOADED,
        db_connected  = True,
        llm_enabled   = LLM_ENABLED,
    )


# ── Metadata ───────────────────────────────────────────────────────────────
@app.get("/metadata", response_model=MetadataOut, tags=["System"])
def metadata():
    return MetadataOut(
        categories    = sorted(CAT_MODEL.classes_.tolist()),
        priorities    = sorted(PRI_MODEL.classes_.tolist()),
        sentiments    = sorted(SEN_MODEL.classes_.tolist()),
        model_version = "v2.0.0",
        encoder       = "all-MiniLM-L6-v2",
    )


# ── Predict (single ticket) ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOut, tags=["Prediction"])
def predict(
    ticket: TicketIn,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    text = _build_text(ticket.Subject, ticket.Description)

    t0      = time.perf_counter()
    result  = _predict_all(text)
    similar = _find_similar(text, top_k=3)

    # LLM suggested reply (non-blocking — returns None if key not set)
    suggested_reply = generate_suggested_reply(
        subject     = ticket.Subject,
        description = ticket.Description,
        category    = result["category"],
        priority    = result["priority"],
        sentiment   = result["sentiment"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Log to SQLite
    log_prediction(
        db             = db,
        subject        = ticket.Subject,
        description    = ticket.Description,
        pred_category  = result["category"],
        pred_priority  = result["priority"],
        pred_sentiment = result["sentiment"],
        confidence     = result["confidence"],
        latency_ms     = latency_ms,
    )

    return PredictionOut(
        predicted_category  = result["category"],
        predicted_priority  = result["priority"],
        predicted_sentiment = result["sentiment"],
        confidence          = round(result["confidence"], 4),
        requires_review     = result["confidence"] < 0.60,
        suggested_reply     = suggested_reply,
        similar_tickets     = similar,
        latency_ms          = round(latency_ms, 3),
    )


# ── Predict batch ──────────────────────────────────────────────────────────
@app.post("/predict_batch", response_model=BatchOut, tags=["Prediction"])
def predict_batch(
    batch: BatchIn,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    texts = [_build_text(t.Subject, t.Description) for t in batch.tickets]

    t0         = time.perf_counter()
    embeddings = ENCODER.encode(texts)

    cat_proba   = CAT_MODEL.predict_proba(embeddings)
    cat_indices = np.argmax(cat_proba, axis=1)
    categories  = [CAT_MODEL.classes_[i] for i in cat_indices]
    confidences = [float(cat_proba[i, cat_indices[i]]) for i in range(len(texts))]

    priorities  = PRI_MODEL.predict(embeddings).tolist()
    sentiments  = SEN_MODEL.predict(embeddings).tolist()

    avg_latency = (time.perf_counter() - t0) * 1000.0 / max(len(texts), 1)

    predictions = []
    for i, ticket in enumerate(batch.tickets):
        log_prediction(
            db             = db,
            subject        = ticket.Subject,
            description    = ticket.Description,
            pred_category  = categories[i],
            pred_priority  = str(priorities[i]),
            pred_sentiment = str(sentiments[i]),
            confidence     = confidences[i],
            latency_ms     = avg_latency,
        )
        predictions.append(BatchPredictionItem(
            predicted_category  = categories[i],
            predicted_priority  = str(priorities[i]),
            predicted_sentiment = str(sentiments[i]),
            confidence          = round(confidences[i], 4),
            requires_review     = confidences[i] < 0.60,
        ))

    return BatchOut(
        predictions    = predictions,
        avg_latency_ms = round(avg_latency, 3),
        count          = len(texts),
    )


# ── Similar tickets ────────────────────────────────────────────────────────
@app.post("/similar", tags=["Prediction"])
def similar(
    ticket: SimilarIn,
    _: str = Depends(verify_api_key),
):
    text    = _build_text(ticket.Subject, ticket.Description)
    results = _find_similar(text, top_k=ticket.top_k)
    return {"similar_tickets": results, "count": len(results)}


# ── Analytics ──────────────────────────────────────────────────────────────
@app.get("/analytics", response_model=AnalyticsOut, tags=["Dashboard"])
def analytics(db: Session = Depends(get_db)):
    return get_analytics(db)


# ── Review queue ───────────────────────────────────────────────────────────
@app.get("/review", response_model=ReviewOut, tags=["Dashboard"])
def review_queue(db: Session = Depends(get_db)):
    queue = get_review_queue(db)
    return ReviewOut(queue=queue, count=len(queue))


@app.post("/review/mark-reviewed", tags=["Dashboard"])
def mark_ticket_reviewed(
    action: ReviewActionIn,
    db: Session = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    success = mark_reviewed(db, action.ticket_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Ticket ID {action.ticket_id} not found.")
    return {"message": f"Ticket {action.ticket_id} marked as reviewed."}

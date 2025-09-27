# api/app.py
import time
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------- Configuration ----------------
# Adjust this path if the repository moves
PROJECT_ROOT = Path(r"D:\customer-support-auto-triage").resolve()
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_model.joblib"

# ---------------- Schemas ----------------
class TicketIn(BaseModel):
    Ticket_ID: int | None = None
    Subject: str = Field(..., min_length=1)
    Description: str = Field(..., min_length=1)
    Priority: str | None = None
    Timestamp: str | None = None

class PredictionOut(BaseModel):
    predicted_category: str
    latency_ms: float

class BatchIn(BaseModel):
    tickets: List[TicketIn]

# ---------------- App & Model ----------------
app = FastAPI(title="Customer Support Ticket Auto‑Triage", version="1.0.0")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")
MODEL = joblib.load(MODEL_PATH)

# ---------------- Homepage ----------------
@app.get("/", response_class=HTMLResponse)
def homepage():
    html = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Customer Support Auto‑Triage</title>
      <style>
        body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;background:#0f172a;color:#e5e7eb}
        .wrap{max-width:860px;margin:56px auto;padding:0 20px}
        .card{border:1px solid #1f2937;border-radius:14px;background:#0b1220;padding:24px}
        h1{margin:0 0 8px 0;font-weight:800}
        p.m{color:#94a3b8;margin:6px 0 16px 0}
        .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
        .box{border:1px solid #1f2937;border-radius:12px;background:#0b1220;padding:14px}
        a.btn{display:inline-block;padding:10px 14px;border-radius:10px;border:1px solid #1f2937;background:#0c1628;color:#e5e7eb;text-decoration:none}
        code{background:#0c1628;border:1px solid #1f2937;padding:2px 6px;border-radius:6px}
        footer{margin-top:14px;color:#94a3b8;font-size:14px}
        @media(max-width:700px){.row{grid-template-columns:1fr}}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="card">
          <h1>Customer Support Auto‑Triage</h1>
          <p class="m">Classify a support ticket into a predefined category in real time.</p>

          <div class="row">
            <div class="box">
              <h3>Start here</h3>
              <p class="m">Open the interactive API docs and try requests in the browser.</p>
              <p><a class="btn" href="/docs">Open Swagger UI</a></p>
              <ul>
                <li><code>GET /health</code> – service status</li>
                <li><code>GET /metadata</code> – labels and vectorizer</li>
                <li><code>POST /predict</code> – classify one ticket</li>
                <li><code>POST /predict_batch</code> – classify many</li>
              </ul>
            </div>
            <div class="box">
              <h3>Quick test (CMD)</h3>
              <p class="m">Paste into Windows Command Prompt while the server is running:</p>
              <p><code>curl -X POST http://127.0.0.1:8000/predict ^</code></p>
              <p><code> -H "Content-Type: application/json" ^</code></p>
              <p><code> -d "{'{'}\\"Subject\\":\\"Login failed\\",\\"Description\\":\\"500 error after update\\"{'}'}"</code></p>
            </div>
          </div>

          <footer>Model is preloaded for low latency. See README and REPORT for metrics and dataset notes.</footer>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

# ---------------- API Routes ----------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH)}

@app.get("/metadata")
def metadata():
    labels = list(getattr(MODEL, "classes_", []))
    vec = MODEL.named_steps.get("tfidf", None)
    vec_info = {
        "type": "TfidfVectorizer",
        "ngram_range": getattr(vec, "ngram_range", None),
        "min_df": getattr(vec, "min_df", None),
        "max_df": getattr(vec, "max_df", None),
        "sublinear_tf": getattr(vec, "sublinear_tf", None),
    } if vec else {}
    return {"labels": labels, "model_version": "v1.0.0", "vectorizer": vec_info}

@app.post("/predict", response_model=PredictionOut)
def predict(ticket: TicketIn):
    if not ticket.Subject or not ticket.Description:
        raise HTTPException(status_code=400, detail="Subject and Description are required")
    text = f"{ticket.Subject} {ticket.Description}".lower().strip()
    t0 = time.perf_counter()
    pred = MODEL.predict([text])[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return PredictionOut(predicted_category=str(pred), latency_ms=round(latency_ms, 3))

@app.post("/predict_batch")
def predict_batch(batch: BatchIn):
    if not batch.tickets:
        raise HTTPException(status_code=400, detail="tickets list cannot be empty")
    texts = [(f"{t.Subject} {t.Description}").lower().strip() for t in batch.tickets]
    t0 = time.perf_counter()
    preds = MODEL.predict(texts)
    avg_latency = (time.perf_counter() - t0) * 1000.0 / max(len(texts), 1)
    return {"predictions": [str(p) for p in preds], "avg_latency_ms": round(avg_latency, 3), "count": len(texts)}
# Customer Support Ticket Auto‑Triage

A small, reproducible project that classifies customer support tickets into predefined categories and serves real‑time predictions via a FastAPI endpoint.

## Overview
This project demonstrates an end‑to‑end workflow: a synthetic dataset, a simple but effective TF‑IDF + Logistic Regression model trained in a notebook, saved artifacts for reproducibility, and a FastAPI server for real‑time inference with latency reporting.

## Dataset
- File: data/customer_support_tickets.csv
- Columns: Ticket_ID, Subject, Description, Category, Priority, Timestamp
- Target: Category (5 classes)
- Nature: Synthetic (programmatically generated to mimic real tickets; safe to share and reproducible)

## Project structure
```
customer-support-auto-triage/
├─ api/
│  ├─ app.py # FastAPI app (homepage + /health, /metadata, /predict, /predict_batch)
│  └─ schemas.py # Pydantic request/response schemas used by the API
├─ data/
│  └─ customer_support_tickets.csv
├─ models/
│  └─ ticket_model.joblib # Trained pipeline: TF‑IDF + LogisticRegression
├─ notebooks/
│  └─ exploration.ipynb # One‑cell training & evaluation; writes artifacts
├─ reports/
│  ├─ REPORT.md # Human‑readable summary
│  ├─ report.json # Metrics (JSON)
│  ├─ test_predictions.csv # True vs predicted on the test split
│  └─ confusion_matrix.png # Saved plot from the notebook
├─ requirements.txt
└─ README.md
```

## Quick start (Windows)
1) Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

2) Install dependencies
```
pip install -r requirements.txt
```

3) Verify the dataset
- Ensure `data/customer_support_tickets.csv` exists.

## Train and evaluate (Notebook)
- Open `notebooks/exploration.ipynb` and run all cells.
- The notebook:
  - Builds text from Subject + Description (lowercased, minimal cleaning).
  - Splits data with stratification, trains TF‑IDF (1–2 grams) + Logistic Regression.
  - Prints metrics, draws and saves a confusion matrix, and writes artifacts.

- Artifacts produced:
  - models/ticket_model.joblib
  - reports/report.json
  - reports/test_predictions.csv
  - reports/confusion_matrix.png

## Final metrics (held‑out test split)
- Accuracy: 1.0
- Precision (macro): 1.0
- Recall (macro): 1.0
- F1 (macro): 1.0
- Avg latency per ticket (ms): 0.0649

Notes on metrics and latency:
- The dataset is synthetic and intentionally separable; expect lower (but still strong) performance on real tickets.
- Latency is measured locally with the model preloaded; values will vary slightly across machines.

## Serve the API (FastAPI)
1) Start the server
```
uvicorn api.app:app --reload
```

2) Use in a browser
- Homepage: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs

3) Sample request (Windows Command Prompt)
```
curl -X POST http://127.0.0.1:8000/predict ^
-H "Content-Type: application/json" ^
-d "{\"Subject\":\"Login failed\",\"Description\":\"500 error after update\"}"
```

## API reference
- GET `/` — Minimal homepage with links and example payload.
- GET `/health` — Service and model status.
- GET `/metadata` — Label set and vectorizer details.
- POST `/predict` — Classify one ticket. Response:
```
{
  "predicted_category": "Technical Issue",
  "latency_ms": 5.7
}
```

- POST `/predict_batch` — Classify multiple tickets. Request:
```
{
  "tickets": [
    {"Subject": "App crash", "Description": "Crashes on login"},
    {"Subject": "Invoice edit", "Description": "Need GST number added"}
  ]
}
```

Response:
```
{
  "predictions": ["Technical Issue", "Billing Inquiry"],
  "avg_latency_ms": 3.9,
  "count": 2
}
```

## Model details
- Features: TF‑IDF with ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True
- Classifier: LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced")
- Saved artifact: models/ticket_model.joblib

## Reproducibility
- requirements.txt pins stable versions tested locally.
- For exact replication, an optional lock file can be generated with:
```
pip freeze > requirements-lock.txt
```
- The notebook and API use project‑relative paths; artifacts are written to models/ and reports/ for clean reruns.

## Troubleshooting
- 404 at “/”: Use /health, /metadata, /predict, or open /docs for Swagger UI.
- Path issues on Windows: Open the project at the repository root; the notebook resolves ROOT automatically.
- Virtual environment permission error: Close terminals and VS Code, kill stray python/uvicorn processes, then reactivate. Recreating the venv is often fastest:
```
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## License and data note
- For classroom/assessment use. The dataset is synthetic and safe to share. Replace with real tickets only in environments that protect sensitive data.

## Author
- Shubham Kumar
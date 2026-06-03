# Project Report — Customer Support Ticket Auto-Triage

**Course:** Capstone Project 2 (Final Year)  
**Version:** 2.0  
**API:** https://customer-support-auto-triage-production.up.railway.app  
**Frontend:** https://customer-support-auto-triage-yfocjabwpgfamvahwzn4xu.streamlit.app

---

## 1. Problem Statement

Customer support teams at software companies receive hundreds of tickets daily. Manually reading, categorising, and routing each ticket wastes agent time and slows down response. Tickets that need urgent attention often sit in a queue alongside low-priority requests because there is no automated way to separate them.

This project builds an automated triage system that classifies incoming tickets, scores their urgency, reads the customer's emotional tone, and drafts a suggested reply — all within a single API call. The goal is to reduce the time a support agent spends on initial ticket assessment from several minutes to under five seconds.

---

## 2. System Overview

The system has three layers:

**ML layer** — three logistic regression classifiers trained on sentence embeddings predict category, priority, and sentiment simultaneously for each incoming ticket.

**API layer** — a FastAPI backend serves predictions, logs every request to a SQLite database, flags low-confidence predictions for human review, and exposes analytics endpoints for the dashboard.

**LLM layer** — after classification, a Groq API call passes the ticket text and predicted labels to `llama-3.1-8b-instant`, which drafts a context-aware suggested reply for the support agent.

---

## 3. Dataset

One of the key findings from this project was that the choice of training data directly determines whether model performance is honest or misleading. Three datasets were used:

**Phase 1 dataset** — 200 rows of clean synthetic data from Kaggle. Ticket descriptions were short and unambiguous, categories were clearly separable, and there was no noise of any kind. This dataset is useful as a baseline but does not represent real customer behaviour.

**Phase 2 dataset** — 485 rows of realistic data generated specifically for this project. Each ticket has a full sentence-length description averaging 134 characters. The dataset includes 15 genuinely ambiguous mixed-intent tickets where the subject and description point toward different categories. Subject lines are varied and non-repetitive. This dataset is a more honest representation of what a real support inbox looks like.

**Phase 3 dataset** — 685 rows combining both datasets above. Used for final model training. Adding the Sentiment column from Phase 2 to Phase 1 rows allowed training a sentiment classifier across the full dataset.

---

## 4. Experiments and Results

Three training phases were run to demonstrate the relationship between data quality, model choice, and performance.

### Phase 1 — TF-IDF on clean data

| Metric | Score |
|--------|-------|
| Accuracy | 1.0000 |
| F1 Macro | 1.0000 |

Perfect accuracy on the clean synthetic dataset. This is expected — the data was too easy. Short, unambiguous descriptions with clearly distinct vocabulary per category meant TF-IDF keyword matching was sufficient to achieve zero errors. This result is not meaningful as a measure of real-world performance.

### Phase 2 — TF-IDF on realistic data

| Metric | Score |
|--------|-------|
| Accuracy | 0.9794 |
| F1 Macro | 0.9797 |

Training on the realistic dataset with the same TF-IDF + LogReg pipeline produced 97.94% accuracy. The drop from 100% shows the model encountering genuinely harder tickets — longer descriptions, overlapping vocabulary between categories, and ambiguous intent. This is a more credible result.

### Phase 3 — Sentence-Transformers on combined data

| Model | Accuracy | F1 Macro |
|-------|----------|----------|
| Category (all-MiniLM-L6-v2 + LogReg) | 0.9781 | 0.9779 |
| Priority (all-MiniLM-L6-v2 + LogReg) | 0.3504 | 0.3436 |
| Sentiment (all-MiniLM-L6-v2 + LogReg) | 0.5255 | 0.5036 |

The category classifier trained on sentence embeddings achieved 97.81% accuracy on the combined dataset — comparable to Phase 2 TF-IDF performance but with genuine semantic understanding rather than keyword matching. The model correctly handles tickets where the subject line contradicts the description body, which TF-IDF cannot do.

Priority and sentiment classifiers show lower accuracy (35% and 52% respectively). This is expected and honest. Priority and sentiment labels in the training data were assigned programmatically based on heuristics rather than human annotation, which means the ground truth itself has noise. These classifiers provide a useful signal but should be treated as indicative rather than definitive — which is why the system routes low-confidence predictions to a human review queue rather than acting on them automatically.

### Summary comparison

| Phase | Dataset | Model | Category Accuracy |
|-------|---------|-------|-------------------|
| Phase 1 | 200 rows (clean) | TF-IDF + LogReg | 100% — not trustworthy |
| Phase 2 | 485 rows (realistic) | TF-IDF + LogReg | 97.94% — honest |
| Phase 3 | 685 rows (combined) | Sentence-Transformers + LogReg | 97.81% — semantic understanding |

---

## 5. API Design

The API was designed to reflect how a real production support triage system would be built, not just how a student project would be built.

**Authentication** — all write endpoints require an API key passed in the `X-API-Key` header. This is the standard pattern used by production APIs.

**Multi-output prediction** — a single `/predict` call returns category, priority, sentiment, confidence, similar tickets, and a suggested reply. Keeping this in one call reduces latency for any frontend or integration consuming the API.

**Confidence-based routing** — predictions below 60% confidence automatically set `requires_review: true` in the response and log the ticket to a review queue. This implements the active learning pattern used by real ML systems — uncertain predictions are surfaced for human correction rather than silently acted upon.

**Prediction logging** — every call to `/predict` and `/predict_batch` writes to a SQLite database. The `/analytics` endpoint reads from this log to return ticket volume, category distribution, sentiment breakdown, and average confidence over time. This makes the system self-monitoring without any additional infrastructure.

**Similar ticket lookup** — the `/similar` endpoint uses TF-IDF cosine similarity against the full training corpus to return the three most semantically similar past tickets. This is a lightweight retrieval-augmented generation pattern that gives agents useful context without requiring a vector database.

---

## 6. LLM Integration

After classification, the ticket text and predicted labels are passed to Groq's `llama-3.1-8b-instant` model with a structured prompt that includes the predicted category, priority, and sentiment as context. The model returns a 3-5 sentence suggested reply tailored to the ticket's content and tone.

The LLM call is non-blocking — if the API key is not configured or the call fails for any reason, the rest of the `/predict` response is returned normally with `suggested_reply: null`. This means the system degrades gracefully rather than returning an error.

Using the predicted category as context in the prompt prevents the LLM from generating generic replies. A billing ticket gets a different reply structure than a bug report, which gets a different structure than a feature request.

---

## 7. Frontend

The frontend is a Streamlit application with four pages:

**Classify Ticket** — the main interface. A support agent pastes a ticket subject and description, hits classify, and sees the predicted category, priority, sentiment, confidence score, similar past tickets, and the LLM-generated suggested reply. Low-confidence predictions show a warning banner.

**Analytics** — pulls from the `/analytics` endpoint and renders four KPI cards (total tickets, average confidence, flagged count, average latency) plus charts for category distribution, sentiment breakdown, priority distribution, and ticket volume over time.

**Review Queue** — shows all tickets flagged for human review with their predicted labels and confidence scores. Agents can expand each ticket and mark it as reviewed.

**Batch Upload** — accepts a CSV file with Subject and Description columns, sends it to `/predict_batch`, and returns a results table with all predictions. Results can be downloaded as a CSV.

---

## 8. Deployment

The backend is deployed on Railway using Docker. The Dockerfile pre-downloads the `all-MiniLM-L6-v2` model during the build step so the container starts immediately without any download on first request. Environment variables are managed through Railway's variable configuration and never stored in the repository.

The frontend is deployed on Streamlit Cloud with a separate minimal `requirements.txt` containing only the four packages the frontend needs — this avoids memory exhaustion during the build step that occurs when trying to install the full ML stack on Streamlit Cloud's free tier.

---

## 9. Limitations and Future Work

**Priority and sentiment accuracy** are the most obvious areas for improvement. The current labels were assigned programmatically — training on human-annotated labels would significantly improve both classifiers.

**Dataset size** — 685 rows is sufficient to demonstrate the system but small by production standards. A real deployment would need thousands of labelled tickets, ideally from the actual support inbox it is being deployed on.

**Similar ticket retrieval** uses TF-IDF cosine similarity, which works well but does not use semantic embeddings. Replacing this with a proper vector database (FAISS or Chroma) storing embeddings of all past tickets would improve retrieval quality, especially for tickets with unusual phrasing.

**Review queue feedback loop** — currently the review queue is a read-only display. In a production system, agent corrections on flagged tickets would feed back into periodic model retraining, gradually improving accuracy on the ticket types the model finds hardest.

---

## 10. Reproduction

To reproduce all results:

```bash
git clone https://github.com/byte-Shubham/customer-support-auto-triage.git
cd customer-support-auto-triage
pip install -r requirements.txt
# Open notebooks/exploration.ipynb and run all cells
# All model artifacts and reports are generated automatically
uvicorn api.app:app --reload
```

All three datasets are included in `data/`. The notebook runs all three phases sequentially and saves confusion matrices, `report.json`, and `test_predictions.csv` to `reports/`.

---

*Capstone Project 2 — Final Year*

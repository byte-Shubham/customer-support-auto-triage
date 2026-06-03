# Customer Support Ticket Auto-Triage

A machine learning system that classifies incoming customer support tickets by category, priority, and sentiment — and drafts a suggested reply using an LLM. Built as a final-year capstone project.

**Live Demo** → [https://customer-support-auto-triage-yfocjabwpgfamvahwzn4xu.streamlit.app](https://customer-support-auto-triage-yfocjabwpgfamvahwzn4xu.streamlit.app)  
**API Docs** → [https://customer-support-auto-triage-production.up.railway.app/docs](https://customer-support-auto-triage-production.up.railway.app/docs)

---

## What it does

When a support ticket comes in, the system:

1. Encodes the ticket text using `all-MiniLM-L6-v2` sentence embeddings
2. Predicts the **category** (Technical Issue, Billing Inquiry, Account Management, Bug Report, Feature Request)
3. Predicts **priority** (High, Medium, Low) and **sentiment** (Positive, Neutral, Negative) in parallel
4. Returns a **confidence score** — tickets below 60% confidence are flagged for human review
5. Finds the **3 most similar past tickets** from the training corpus using TF-IDF cosine similarity
6. Generates a **suggested agent reply** using Groq's `llama-3.1-8b-instant` model

Everything is served through a FastAPI backend with API key authentication, SQLite prediction logging, and a live analytics endpoint.

---

## Project structure

```
customer-support-auto-triage/
├── api/
│   ├── app.py          # FastAPI app — all endpoints
│   ├── schemas.py      # Pydantic request/response models
│   ├── database.py     # SQLite logging via SQLAlchemy
│   ├── auth.py         # API key authentication
│   └── llm.py          # Groq LLM reply generation
├── data/
│   ├── customer_support_tickets.csv           # Phase 1 — original 200 rows
│   ├── customer_support_dataset_500_fixed.csv # Phase 2 — realistic 485 rows
│   └── customer_support_combined.csv          # Phase 3 — combined 685 rows
├── models/
│   ├── category_model.joblib   # Category classifier
│   ├── priority_model.joblib   # Priority classifier
│   ├── sentiment_model.joblib  # Sentiment classifier
│   ├── tfidf_index.joblib      # TF-IDF index for /similar
│   └── training_corpus.csv     # Corpus for similarity lookup
├── notebooks/
│   └── exploration.ipynb       # 3-phase training experiment
├── frontend/
│   ├── app.py                  # Streamlit frontend
│   └── requirements.txt        # Frontend-only dependencies
├── reports/
│   ├── confusion_matrix_phase1.png
│   ├── confusion_matrix_phase2.png
│   ├── confusion_matrix_phase3_category.png
│   ├── report.json
│   └── test_predictions.csv
├── Dockerfile
├── docker-compose.yml
├── railway.json
└── requirements.txt
```

---

## API endpoints

All POST routes require the header `X-API-Key: <your-key>`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Homepage |
| GET | `/health` | Service and model status |
| GET | `/metadata` | Categories, priorities, sentiments, model info |
| POST | `/predict` | Classify a single ticket — returns category, priority, sentiment, confidence, similar tickets, suggested reply |
| POST | `/predict_batch` | Classify multiple tickets from a list |
| POST | `/similar` | Find similar past tickets for a given ticket |
| GET | `/analytics` | Ticket volume, category distribution, sentiment breakdown, avg confidence |
| GET | `/review` | Low-confidence tickets pending human review |
| POST | `/review/mark-reviewed` | Mark a review queue ticket as resolved |

---

## Quick start

**Requirements:** Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/byte-Shubham/customer-support-auto-triage.git
cd customer-support-auto-triage

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file in project root
# GROQ_API_KEY=your-groq-key-from-console.groq.com
# API_KEY=your-chosen-api-key

# 5. Run the notebook to generate model files
# Open notebooks/exploration.ipynb and run all cells

# 6. Start the API
uvicorn api.app:app --reload

# 7. Start the frontend (new terminal)
streamlit run frontend/app.py
```

API runs at `http://127.0.0.1:8000` — open `/docs` for the interactive Swagger UI.  
Frontend runs at `http://localhost:8501`.

---

## Run with Docker

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`.

---

## Model details

The system uses three separate `LogisticRegression` classifiers, each trained on 384-dimensional sentence embeddings from `all-MiniLM-L6-v2`:

| Model | Task | Classes |
|-------|------|---------|
| category_model | Ticket category | 5 classes |
| priority_model | Ticket priority | 3 classes (High, Medium, Low) |
| sentiment_model | Customer sentiment | 3 classes (Positive, Neutral, Negative) |

Similarity search uses a TF-IDF vectorizer with cosine similarity against the full training corpus.

---

## Dataset

Three datasets were used across the three training phases:

| Dataset | Rows | Description |
|---------|------|-------------|
| customer_support_tickets.csv | 200 | Clean synthetic dataset from Kaggle |
| customer_support_dataset_500_fixed.csv | 485 | Realistic dataset with sentence-length descriptions, varied subjects, and 15 genuinely ambiguous mixed-intent tickets |
| customer_support_combined.csv | 685 | Combined dataset used for final model training |

The fixed dataset was generated specifically for this project to address the limitations of the original clean synthetic data.

---

## Deployment

| Service | Purpose | URL |
|---------|---------|-----|
| Railway | FastAPI backend | [API](https://customer-support-auto-triage-production.up.railway.app) |
| Streamlit Cloud | Frontend UI | [App](https://customer-support-auto-triage-yfocjabwpgfamvahwzn4xu.streamlit.app) |

---

## Team

Built by a 5-member group as part of the final-year Capstone Project 2.

| Role | Responsibility |
|------|---------------|
| ML + Backend | Model training, FastAPI, database, LLM integration |
| Frontend | Streamlit dashboard and UI |
| DevOps | Docker, Railway deployment |
| Evaluation | Metrics, confusion matrices, report |
| Data + Testing | Dataset preparation, API testing |

---

## License

For academic and assessment use. The datasets used are synthetic and safe to share publicly.

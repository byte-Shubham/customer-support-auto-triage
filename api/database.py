# api/database.py
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String,
    create_engine, func
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── DB file sits at project_root/data/predictions.db ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "predictions.db"
ENGINE       = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)


# ── ORM base ───────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Prediction log table ───────────────────────────────────────────────────
class PredictionLog(Base):
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, index=True)
    subject          = Column(String,  nullable=False)
    description      = Column(String,  nullable=False)
    pred_category    = Column(String,  nullable=False)
    pred_priority    = Column(String,  nullable=False)
    pred_sentiment   = Column(String,  nullable=False)
    confidence       = Column(Float,   nullable=False)   # category confidence (0–1)
    latency_ms       = Column(Float,   nullable=False)
    requires_review  = Column(Boolean, default=False)    # True if confidence < 0.60
    reviewed         = Column(Boolean, default=False)    # agent marked as reviewed
    timestamp        = Column(DateTime, default=datetime.utcnow)


# ── Create tables on import ────────────────────────────────────────────────
def init_db() -> None:
    Base.metadata.create_all(bind=ENGINE)


# ── FastAPI dependency ─────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Write helpers ──────────────────────────────────────────────────────────
def log_prediction(db: Session, subject: str, description: str,
                   pred_category: str, pred_priority: str,
                   pred_sentiment: str, confidence: float,
                   latency_ms: float) -> PredictionLog:
    entry = PredictionLog(
        subject         = subject,
        description     = description,
        pred_category   = pred_category,
        pred_priority   = pred_priority,
        pred_sentiment  = pred_sentiment,
        confidence      = round(confidence, 4),
        latency_ms      = round(latency_ms, 3),
        requires_review = confidence < 0.60,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def mark_reviewed(db: Session, ticket_id: int) -> bool:
    entry = db.query(PredictionLog).filter(PredictionLog.id == ticket_id).first()
    if not entry:
        return False
    entry.reviewed = True
    db.commit()
    return True


# ── Read helpers (used by /analytics and /review) ─────────────────────────
def get_analytics(db: Session) -> dict:
    total = db.query(func.count(PredictionLog.id)).scalar() or 0

    # category distribution
    cat_rows = (
        db.query(PredictionLog.pred_category, func.count(PredictionLog.id))
        .group_by(PredictionLog.pred_category).all()
    )
    category_dist = {row[0]: row[1] for row in cat_rows}

    # sentiment distribution
    sen_rows = (
        db.query(PredictionLog.pred_sentiment, func.count(PredictionLog.id))
        .group_by(PredictionLog.pred_sentiment).all()
    )
    sentiment_dist = {row[0]: row[1] for row in sen_rows}

    # priority distribution
    pri_rows = (
        db.query(PredictionLog.pred_priority, func.count(PredictionLog.id))
        .group_by(PredictionLog.pred_priority).all()
    )
    priority_dist = {row[0]: row[1] for row in pri_rows}

    # averages
    avg_conf    = db.query(func.avg(PredictionLog.confidence)).scalar() or 0.0
    avg_latency = db.query(func.avg(PredictionLog.latency_ms)).scalar() or 0.0
    flagged     = db.query(func.count(PredictionLog.id)).filter(
                      PredictionLog.requires_review == True).scalar() or 0

    # tickets over time (date → count)
    time_rows = (
        db.query(
            func.strftime("%Y-%m-%d", PredictionLog.timestamp).label("date"),
            func.count(PredictionLog.id).label("count")
        )
        .group_by("date")
        .order_by("date")
        .all()
    )
    tickets_over_time = [{"date": row[0], "count": row[1]} for row in time_rows]

    return {
        "total_tickets":     total,
        "category_dist":     category_dist,
        "sentiment_dist":    sentiment_dist,
        "priority_dist":     priority_dist,
        "avg_confidence":    round(float(avg_conf), 4),
        "avg_latency_ms":    round(float(avg_latency), 3),
        "flagged_for_review": flagged,
        "tickets_over_time": tickets_over_time,
    }


def get_review_queue(db: Session, limit: int = 50) -> list:
    rows = (
        db.query(PredictionLog)
        .filter(PredictionLog.requires_review == True,
                PredictionLog.reviewed == False)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id":             r.id,
            "subject":        r.subject,
            "description":    r.description,
            "pred_category":  r.pred_category,
            "pred_priority":  r.pred_priority,
            "pred_sentiment": r.pred_sentiment,
            "confidence":     r.confidence,
            "timestamp":      r.timestamp.isoformat(),
        }
        for r in rows
    ]

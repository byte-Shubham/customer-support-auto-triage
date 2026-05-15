# api/schemas.py
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ── Request schemas ────────────────────────────────────────────────────────

class TicketIn(BaseModel):
    Ticket_ID:   Optional[int] = None
    Subject:     str = Field(..., min_length=1, description="Ticket subject line")
    Description: str = Field(..., min_length=1, description="Full ticket description")
    Priority:    Optional[str] = None
    Timestamp:   Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "Subject": "Login failed after update",
                "Description": "I can't log in since the last app update. It shows a 500 error on the login screen."
            }
        }
    }


class BatchIn(BaseModel):
    tickets: List[TicketIn] = Field(..., min_length=1)


class SimilarIn(BaseModel):
    Subject:     str = Field(..., min_length=1)
    Description: str = Field(..., min_length=1)
    top_k:       int = Field(default=3, ge=1, le=10)


class ReviewActionIn(BaseModel):
    ticket_id: int = Field(..., description="ID from the review queue to mark as reviewed")


# ── Response schemas ───────────────────────────────────────────────────────

class SimilarTicket(BaseModel):
    subject:   str
    category:  str
    score:     float = Field(..., description="Cosine similarity score (0–1)")


class PredictionOut(BaseModel):
    predicted_category: str
    predicted_priority: str
    predicted_sentiment: str
    confidence:          float = Field(..., description="Category model confidence (0–1)")
    requires_review:     bool  = Field(..., description="True if confidence < 0.60")
    suggested_reply:     Optional[str] = Field(None, description="LLM-generated reply draft")
    similar_tickets:     List[SimilarTicket] = Field(default_factory=list)
    latency_ms:          float


class BatchPredictionItem(BaseModel):
    predicted_category:  str
    predicted_priority:  str
    predicted_sentiment: str
    confidence:          float
    requires_review:     bool


class BatchOut(BaseModel):
    predictions:     List[BatchPredictionItem]
    avg_latency_ms:  float
    count:           int


class AnalyticsOut(BaseModel):
    total_tickets:      int
    category_dist:      Dict[str, int]
    sentiment_dist:     Dict[str, int]
    priority_dist:      Dict[str, int]
    avg_confidence:     float
    avg_latency_ms:     float
    flagged_for_review: int
    tickets_over_time:  List[Dict[str, Any]]


class ReviewTicket(BaseModel):
    id:              int
    subject:         str
    description:     str
    pred_category:   str
    pred_priority:   str
    pred_sentiment:  str
    confidence:      float
    timestamp:       str


class ReviewOut(BaseModel):
    queue:  List[ReviewTicket]
    count:  int


class HealthOut(BaseModel):
    status:        str
    models_loaded: bool
    db_connected:  bool
    llm_enabled:   bool


class MetadataOut(BaseModel):
    model_config = {"protected_namespaces": ()}
    categories:    List[str]
    priorities:    List[str]
    sentiments:    List[str]
    model_version: str
    encoder:       str

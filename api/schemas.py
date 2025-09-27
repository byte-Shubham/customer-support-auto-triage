from pydantic import BaseModel, Field
from typing import Optional, List

class TicketIn(BaseModel):
    Ticket_ID: Optional[int] = None
    Subject: str = Field(..., min_length=1)
    Description: str = Field(..., min_length=1)
    Priority: Optional[str] = None
    Timestamp: Optional[str] = None

class PredictionOut(BaseModel):
    predicted_category: str
    latency_ms: float

class BatchIn(BaseModel):
    tickets: List[TicketIn]
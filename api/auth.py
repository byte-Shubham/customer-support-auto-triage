# api/auth.py
import os
from dotenv import load_dotenv
from fastapi import Header, HTTPException, status

load_dotenv()

# Read the API key from .env — falls back to a default for local dev
_API_KEY = os.getenv("API_KEY", "capstone-dev-key-2024")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    FastAPI dependency — add to any route you want to protect:

        @app.post("/predict")
        def predict(ticket: TicketIn, _: str = Depends(verify_api_key)):
            ...

    The client must send the header:  X-API-Key: <your_key>
    Returns 401 if the key is missing or wrong.
    """
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass your key in the X-API-Key header.",
        )
    return x_api_key

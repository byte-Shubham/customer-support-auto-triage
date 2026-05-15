# api/llm.py
import os
from dotenv import load_dotenv

load_dotenv()

_GROQ_KEY = os.getenv("GROQ_API_KEY", "")


def generate_suggested_reply(
    subject: str,
    description: str,
    category: str,
    priority: str,
    sentiment: str,
) -> str | None:
    """
    Calls Groq API (llama-3.1-8b-instant) to generate a suggested
    customer support reply based on the classified ticket.

    Returns a reply string, or None if the API key is not configured
    so the rest of /predict still works without LLM.
    """
    if not _GROQ_KEY:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=_GROQ_KEY)

        system_prompt = (
            "You are a professional customer support agent. "
            "Your job is to write a short, empathetic, and helpful reply "
            "to a customer support ticket. "
            "Keep the reply between 3 and 5 sentences. "
            "Be polite, acknowledge the issue, and give a clear next step. "
            "Do not use placeholders like [Name] or [Order ID] — write naturally. "
            "Do not start with 'Dear Customer' — use 'Hi there' or 'Hello'."
        )

        user_prompt = (
            f"Ticket Category : {category}\n"
            f"Priority        : {priority}\n"
            f"Customer Tone   : {sentiment}\n"
            f"Subject         : {subject}\n"
            f"Description     : {description}\n\n"
            "Write a suggested reply for the support agent to send to this customer."
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # Never crash the main /predict endpoint because of LLM failure
        print(f"[llm.py] LLM call failed: {e}")
        return None

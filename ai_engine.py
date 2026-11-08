"""
ai_engine.py — Gemini-powered text-to-SQL engine for QueryMind AI.

Uses the new `google.genai` SDK (google-genai package).
"""

import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are QueryMind AI, an expert SQL assistant.
Your ONLY job is to convert the user's natural language question into a single, valid SQLite SQL statement.

Rules:
1. Return ONLY the raw SQL statement — no explanations, no markdown code fences, no comments.
2. Use only the tables and columns present in the provided schema.
3. Never generate DROP, ALTER, TRUNCATE, CREATE or RENAME statements.
4. For UPDATE or DELETE, always include a WHERE clause unless the user explicitly says "all rows".
5. Use double-quotes for table/column identifiers if they contain spaces or reserved words.
6. Apply a LIMIT clause (default 100) for SELECT unless the user specifies otherwise.
7. If the question cannot be answered with the given schema, respond with exactly: CANNOT_GENERATE
"""


# ── Conversation memory ────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Lightweight sliding-window conversation history.
    Each turn: {"role": "user"|"assistant", "content": str}
    """

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.history: list[dict] = []

    def add(self, user_msg: str, assistant_msg: str):
        self.history.append({"role": "user",      "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        self.history = self.history[-(self.max_turns * 2):]

    def as_text(self) -> str:
        if not self.history:
            return ""
        lines = ["--- Previous conversation ---"]
        for msg in self.history:
            prefix = "User" if msg["role"] == "user" else "SQL"
            lines.append(f"{prefix}: {msg['content']}")
        lines.append("--- End of previous conversation ---")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()

    def get_last_sql(self) -> str | None:
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None


# ── Main generation function ───────────────────────────────────────────────────

def generate_sql(
    user_question: str,
    schema_context: str,
    memory: ConversationMemory,
) -> str:
    """
    Call Gemini to convert a natural language question into SQL.

    Returns a SQL string or raises RuntimeError on failure.
    """
    client = _get_client()

    history_text = memory.as_text()
    last_sql = memory.get_last_sql()

    # Follow-up pronoun hints
    follow_up_hint = ""
    if last_sql and any(
        w in user_question.lower()
        for w in ["them", "those", "it", "that", "result", "above", "previous"]
    ):
        follow_up_hint = (
            f"\nNote: The user may be referring to the previous query:\n{last_sql}\n"
            "Incorporate it as a subquery or CTE if appropriate.\n"
        )

    prompt = f"""{SYSTEM_PROMPT}

--- Database Schema ---
{schema_context}
--- End Schema ---

{history_text}
{follow_up_hint}
Current question: {user_question}

SQL:"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )

    sql = response.text.strip() if response.text else ""

    # Strip any accidental markdown fences
    sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql).strip()

    return sql


# ── API health check ───────────────────────────────────────────────────────────

def check_api_connection() -> tuple[bool, str]:
    """Verify Gemini API is reachable. Returns (ok, message)."""
    if not GOOGLE_API_KEY:
        return False, "GOOGLE_API_KEY not set."
    try:
        client = _get_client()
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Reply with the single word: OK",
            config=types.GenerateContentConfig(max_output_tokens=10),
        )
        if resp.text:
            return True, f"Connected — model: {GEMINI_MODEL}"
        return False, "Empty response from Gemini API."
    except Exception as e:
        return False, str(e)

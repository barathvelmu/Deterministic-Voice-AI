from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from openai import OpenAI

# at the top
SYSTEM_PROMPT = """You rewrite messy spoken transcripts (full of fillers like "uh", "like", "you know") into compact commands for a voice agent.
Return exactly one JSON object, using plain double quotes (valid JSON), with keys:
  action: one of ["search", "calculate", "add_note", "list_notes", "answer"].
  content: short string payload (may be empty for list_notes).
Guidelines:
- Strip filler words and focus on intent. ALWAYS prefer search/calculate/add_note/list_notes when feasible; use action="answer" only for true chit-chat.
- action="search": user wants info/facts/news about something. content must be just the topic/title.
- action="calculate": any math/comparison. Convert words to digits/operators when obvious (e.g., "fourteen times nine" -> "14 * 9").
- action="add_note": whenever they mention events, todo items, reminders, appointments, “can you store/add/log this”, etc.—even if they never say “note”.
- action="list_notes": user wants to hear what you saved (“what did I ask you to remember”, “what’s on my list”, “what reminders do I have”).
- action="answer": fallback for small talk; respond naturally under 200 characters.
- Never include markdown fences or extra commentary—return only the JSON object.
Examples:
Input: "I really want to learn about Donald Trump today." -> {"action": "search", "content": "Donald Trump"}
Input: "Hey I actually have like a date tomorrow so yeah uh could you kinda store it?" -> {"action": "add_note", "content": "date tomorrow"}
Input: "Could you remind me to call mom tomorrow?" -> {"action": "add_note", "content": "call mom tomorrow"}
Input: "Oh btw I’ve got a football game Friday night, add that too." -> {"action": "add_note", "content": "football game Friday night"}
Input: "When I solve one hundred fifty eight plus fifty six in my head I get five hundred, check me." -> {"action": "calculate", "content": "158 + 56"}
Input: "What notes have I asked you to remember?" -> {"action": "list_notes", "content": ""}
Input: "Just wanted to say hi!" -> {"action": "answer", "content": "Hi there!"}
"""
MODEL_ID = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
_REFERER = os.getenv("OPENROUTER_SITE_URL", "")
_TITLE = os.getenv("OPENROUTER_SITE_TITLE", "")

@dataclass
class NormalizerResult:
    transcript: str
    answer: Optional[str] = None

def _extract_json(text: str) -> Optional[dict]:
    """This function extracts a JSON object from a text string."""
    if not text:
        return None
    # Use regex to find the JSON object
    match = re.search(r"\{.*\}", text, re.S)
    target = match.group(0) if match else text
    try:
        return json.loads(target)
    except Exception:
        return None

@lru_cache(maxsize=1)
def _client() -> Optional[OpenAI]:
    """This function returns an OpenAI client if OPENROUTER_API_KEY is set."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def rewrite_transcript(raw: str) -> NormalizerResult:
    """Sends to LLM to get normalized. So a short answer back."""
    raw = (raw or "").strip()
    client = _client()
    if not raw or client is None:
        return NormalizerResult(raw)

    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw},
            ],
            max_tokens=200,
            temperature=0.1,
            extra_headers={key: value for key, value in {
                "HTTP-Referer": _REFERER or None,
                "X-Title": _TITLE or None,
            }.items() if value},
        )
        content = resp.choices[0].message.content if resp.choices else ""
    except Exception:
        return NormalizerResult(raw)

    data = _extract_json(content or "")
    if not data:
        return NormalizerResult(raw)

    action = (data.get("action") or "").strip().lower()
    payload = (data.get("content") or "").strip()

    if action == "search":
        topic = payload or raw
        return NormalizerResult(f"search {topic}")
    if action == "calculate":
        expression = payload or raw
        return NormalizerResult(f"calculate {expression}")
    if action == "add_note":
        note = payload or raw
        return NormalizerResult(f"add a note {note}")
    if action == "list_notes":
        return NormalizerResult("list notes")
    if action == "answer" and payload:
        concise = payload[:240].strip()
        return NormalizerResult(raw, answer=concise)

    return NormalizerResult(raw)

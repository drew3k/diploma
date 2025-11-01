# app/pipeline/feedback.py
from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

_APP_ROOT = Path(__file__).resolve().parents[1]
_FEED_DIR = _APP_ROOT / "data" / "feedback"
_FEED_DIR.mkdir(parents=True, exist_ok=True)
_FEED_PATH = _FEED_DIR / "candidates.jsonl"


def log_candidates(text: str, spans, source: str = "api") -> None:
    """Пишет кандидатов в data/feedback/candidates.jsonl (по одному JSON в строке)."""
    try:
        item = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "source": source,
            "text": text,
            "spans": [
                {
                    "start": int(s.start),
                    "end": int(s.end),
                    "label": str(s.label),
                    "text": str(s.text),
                }
                for s in (spans or [])
            ],
        }
        with _FEED_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    except Exception as e:
        # чтобы не уронить запрос — просто сообщим в консоль
        print(f"[feedback] failed to write candidates: {e}")

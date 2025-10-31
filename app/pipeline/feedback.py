from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from app.settings import settings
from .utils import Span

_FEED_DIR = settings.data_dir / "feedback"
_FEED_DIR.mkdir(parents=True, exist_ok=True)


def log_candidates(text: str, spans: list[Span], source: str = "web") -> None:
    """Логируем текст и найденные сущности для последующей проверки/разметки."""
    item = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "text": text,
        "spans": [
            {"start": s.start, "end": s.end, "label": s.label, "text": s.text}
            for s in spans
        ],
    }
    path = _FEED_DIR / "candidates.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

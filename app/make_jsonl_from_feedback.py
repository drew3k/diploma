from __future__ import annotations
from pathlib import Path
import json
from sklearn.model_selection import train_test_split  # pip install scikit-learn

ROOT = Path(__file__).resolve().parents[1]
FEED = ROOT / "data" / "feedback" / "candidates.jsonl"
OUT_DIR = ROOT / "data" / "labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP = {"PERSON", "ADDRESS", "LOCATION"}  # LOCATION будет смэплен в ADDRESS в обучалке


def load_feedback(path: Path):
    items = []
    if not path.exists():
        print(f"[warn] {path} not found")
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            spans = obj.get("spans", [])
            # берём только PERSON/ADDRESS (LOCATION оставим, обучалка маппит в ADDRESS)
            ents = []
            for s in spans:
                lbl = str(s.get("label", "")).upper()
                if lbl in KEEP:
                    ents.append([int(s["start"]), int(s["end"]), lbl])
            if text and ents:
                items.append({"text": text, "entities": ents})
    return items


def dump_jsonl(recs, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(recs)} -> {path}")


def main():
    data = load_feedback(FEED)
    if not data:
        print(
            "[info] no data collected yet. Drive the app to populate candidates.jsonl."
        )
        return
    train, dev = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    dump_jsonl(train, OUT_DIR / "train.jsonl")
    dump_jsonl(dev, OUT_DIR / "dev.jsonl")


if __name__ == "__main__":
    main()

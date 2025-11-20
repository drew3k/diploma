from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from pipeline.detect import detect_spans
from pipeline.utils import Span, smart_mask
from metrics import classification_metrics, pretty_print


# Only use fields that our detector supports
FIELD_TO_LABEL = {
    "full_name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "bank": "CREDIT_CARD",
}


@dataclass
class LabeledText:
    text: str
    spans: List[Span]


def _make_snippet(row: dict) -> LabeledText:
    """Build a Russian snippet embedding only detectable fields and record gold spans."""
    parts: list[tuple[str, str | None]] = [
        ("Меня зовут ", None),
        (str(row.get("full_name", "")), FIELD_TO_LABEL["full_name"]),
        (". Моя почта ", None),
        (str(row.get("email", "")), FIELD_TO_LABEL["email"]),
        (". Мой телефон ", None),
        (str(row.get("phone", "")), FIELD_TO_LABEL["phone"]),
        (". Номер карты ", None),
        (str(row.get("bank", "")), FIELD_TO_LABEL["bank"]),
        (".\n", None),
    ]

    spans: list[Span] = []
    buf: list[str] = []
    pos = 0
    for token, lbl in parts:
        if lbl is not None and token:
            start = pos
            end = start + len(token)
            spans.append(Span(token, start, end, lbl))
        buf.append(token)
        pos += len(token)
    return LabeledText(text="".join(buf), spans=spans)


def _apply_policy(text: str, spans: Sequence[Span], policy: str) -> str:
    """Apply redaction policy to text using predicted spans."""
    spans_sorted = sorted(spans, key=lambda s: s.start)
    out: list[str] = []
    i = 0
    for s in spans_sorted:
        out.append(text[i : s.start])
        if policy == "remove":
            repl = ""
        else:
            # default: mask
            repl = smart_mask(s.label, s.text or "")
        out.append(repl)
        i = s.end
    out.append(text[i:])
    return "".join(out)


def _iter_rows(df, limit: int | None) -> Iterable[dict]:
    count = 0
    for _, row in df.iterrows():
        yield row.to_dict()
        count += 1
        if limit and count >= limit:
            break


def train_from_fake(
    n_rows: int = 1000,
    policy: str = "mask",
    languages: list[str] | None = None,
    batch_size: int = 5000,
):
    """
    Build a synthetic labeled corpus from fake.py and evaluate the current detector.

    This does not fine-tune spaCy/Presidio models (out of scope here), but
    provides an end-to-end loop to measure how well current detection handles
    fields it is expected to support.
    """
    from importlib import import_module

    fake = import_module("fake")
    if not hasattr(fake, "make_df"):
        raise RuntimeError("fake.py must expose make_df(n_rows: int) -> DataFrame")

    df = fake.make_df(n_rows)

    langs = (languages or ["ru", "en"])  # restrict to supported languages inside detect_spans
    allowed = set(FIELD_TO_LABEL.values())

    true_spans: list[Span] = []
    pred_spans: list[Span] = []

    # Build and process in batches to avoid huge single-doc processing
    batch: list[LabeledText] = []
    global_offset = 0

    def process_batch(items: list[LabeledText], start_offset: int) -> int:
        if not items:
            return 0
        chunk_text = "".join(x.text for x in items)

        # true spans with offsets
        local_true: list[Span] = []
        local_offset = 0
        for doc in items:
            for s in doc.spans:
                local_true.append(Span(s.text, s.start + local_offset, s.end + local_offset, s.label))
            local_offset += len(doc.text)

        # detect on chunk
        local_pred = detect_spans(chunk_text, langs, allowed)

        # accumulate global spans
        for s in local_true:
            true_spans.append(Span(s.text, s.start + start_offset, s.end + start_offset, s.label))
        for s in local_pred:
            pred_spans.append(Span(s.text, s.start + start_offset, s.end + start_offset, s.label))

        return len(chunk_text)

    for row in _iter_rows(df, limit=n_rows):
        batch.append(_make_snippet(row))
        if len(batch) >= batch_size:
            consumed = process_batch(batch, global_offset)
            global_offset += consumed
            batch.clear()

    if batch:
        consumed = process_batch(batch, global_offset)
        global_offset += consumed
        batch.clear()

    # classification metrics
    cls = classification_metrics(true_spans, pred_spans, allowed)
    return cls


if __name__ == "__main__":
    cls_metrics = train_from_fake(
        n_rows=1000, policy="mask", languages=["ru", "en"], batch_size=5000
    )
    pretty_print(cls_metrics)

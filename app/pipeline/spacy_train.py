from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

from app.settings import settings


@dataclass
class Record:
    text: str
    entities: List[Tuple[int, int, str]]  # (start, end, label)


def _build_ru_records(n_rows: int) -> List[Record]:
    from importlib import import_module
    from .train import _make_snippet

    fake = import_module("fake")
    df = fake.make_df(n_rows=n_rows, ru_share=1.0)

    recs: List[Record] = []
    for _, row in df.iterrows():
        lt = _make_snippet(row.to_dict())
        # Keep only PERSON entities for spaCy training
        ents = [(s.start, s.end, "PERSON") for s in lt.spans if s.label == "PERSON"]
        recs.append(Record(text=lt.text, entities=ents))

    # Add negative heading-like examples to reduce false positives on headings
    heading_words = [
        "Введение",
        "Содержание",
        "Общие",
        "Положения",
        "Раздел",
        "Глава",
        "Статья",
        "Предмет",
        "Договор",
        "Приложение",
        "Описание",
        "Требования",
        "Порядок",
        "Определения",
        "Термины",
    ]
    import random as _rnd
    for _ in range(max(100, n_rows // 10)):
        w1 = _rnd.choice(heading_words)
        w2 = _rnd.choice(heading_words)
        # Ensure some variance and avoid duplicates
        if _rnd.random() < 0.6:
            text = f"{w1} {w2}"
        else:
            w3 = _rnd.choice(heading_words)
            text = f"{w1} {w2} {w3}"
        recs.append(Record(text=text, entities=[]))
    return recs


def _build_en_records(n_rows: int) -> List[Record]:
    # Simple English template for synthetic PERSON
    from faker import Faker

    fk = Faker("en_US")
    recs: List[Record] = []
    for _ in range(n_rows):
        name = fk.name()
        email = fk.email()
        phone = fk.phone_number()
        card = fk.credit_card_number()
        # Compose snippet; only PERSON is annotated for NER
        parts = [
            ("My name is ", None),
            (name, "PERSON"),
            (". Email ", None),
            (email, None),
            (". Phone ", None),
            (phone, None),
            (". Card number ", None),
            (card, None),
            (".\n", None),
        ]
        text = []
        ents: List[Tuple[int, int, str]] = []
        pos = 0
        for tok, lbl in parts:
            if lbl:
                ents.append((pos, pos + len(tok), lbl))
            text.append(tok)
            pos += len(tok)
        recs.append(Record(text="".join(text), entities=ents))
    return recs


def _to_docbin(nlp, records: List[Record]) -> DocBin:
    db = DocBin(store_user_data=False)
    for r in records:
        doc = nlp.make_doc(r.text)
        ents = []
        for start, end, label in r.entities:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db


def _ensure_ner(nlp) -> spacy.language.Language:
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    if "PERSON" not in ner.labels:
        ner.add_label("PERSON")
    return nlp


def train_spacy_person(
    lang: str = "ru",
    n_rows: int = 20000,
    n_iter: int = 10,
    batch_size: int = 256,
    dropout: float = 0.2,
    base_model: str | None = None,
    out_dir: str | None = None,
):
    """
    Fine-tune spaCy NER for PERSON on synthetic data.

    - lang: 'ru' or 'en'
    - base_model: if None, uses ru_core_news_md/en_core_web_md
    - out_dir: if None, saves under settings.data_dir / 'models' / lang
    """
    if base_model is None:
        base_model = "ru_core_news_md" if lang == "ru" else "en_core_web_md"

    print(f"Loading base model: {base_model}")
    nlp = spacy.load(base_model)
    _ensure_ner(nlp)

    print(f"Building {lang} synthetic corpus: {n_rows} examples…")
    recs = _build_ru_records(n_rows) if lang == "ru" else _build_en_records(n_rows)
    random.shuffle(recs)

    # Split train/dev
    split = int(len(recs) * 0.9)
    train_recs, dev_recs = recs[:split], recs[split:]

    # Prepare examples
    train_db = _to_docbin(nlp, train_recs)
    dev_db = _to_docbin(nlp, dev_recs)

    train_docs = list(train_db.get_docs(nlp.vocab))
    dev_docs = list(dev_db.get_docs(nlp.vocab))

    train_examples = []
    for d in train_docs:
        train_examples.append(Example.from_dict(d, {"entities": [(e.start_char, e.end_char, e.label_) for e in d.ents]}))

    # Disable other pipes for faster training
    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()
        for it in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            for i in range(0, len(train_examples), batch_size):
                batch = train_examples[i : i + batch_size]
                nlp.update(batch, sgd=optimizer, drop=dropout, losses=losses)
            print(f"Iter {it+1}/{n_iter} - loss={losses.get('ner', 0):.2f}")

    # Quick eval on dev (exact entity match)
    gold = 0
    pred = 0
    hit = 0
    for d in dev_docs:
        p = nlp(d.text)
        gold_set = {(e.start_char, e.end_char, e.label_) for e in d.ents}
        pred_set = {(e.start_char, e.end_char, e.label_) for e in p.ents if e.label_ == "PERSON"}
        gold += len(gold_set)
        pred += len(pred_set)
        hit += len(gold_set & pred_set)
    prec = hit / pred if pred else 0.0
    rec = hit / gold if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Dev eval PERSON - P={prec:.3f} R={rec:.3f} F1={f1:.3f} (gold={gold} pred={pred} hit={hit})")

    # Save model
    if out_dir is None:
        out_dir = str(settings.data_dir / "models" / ("ru" if lang == "ru" else "en"))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    # Example: train RU PERSON and save under data/models/ru
    train_spacy_person(lang="ru", n_rows=20000, n_iter=6, batch_size=256)

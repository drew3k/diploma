from __future__ import annotations
from pathlib import Path
import json, hashlib, random
from typing import Tuple

from fake import make_df  # твоя генерация синтетики

try:
    from faker import Faker

    _faker = Faker("ru_RU")
except Exception:
    _faker = None

DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _add_field(buf: list[str], prefix: str, value: str) -> Tuple[int, int]:
    cur = "".join(buf)
    start = len(cur) + len(prefix)
    buf.append(prefix)
    buf.append(value)
    end = start + len(value)
    buf.append("\n")
    return start, end


def _1line(s: str | None) -> str:
    s = (s or "").replace("\r\n", ", ").replace("\n", ", ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip(", ").strip()


def _first(row: dict, *keys: str) -> str:
    for k in keys:
        v = row.get(k)
        if v:
            return str(v)
    return ""


def _stable_company(seed_text: str) -> str:
    """
    Детормин. генерация названия компании из seed_text.
    Если Faker доступен — берём его; иначе простой шаблон.
    """
    if not seed_text:
        seed_text = "seed"
    h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(h)
    if _faker:
        # Сделаем несколько попыток, берём первую не пустую
        names = {_faker.company() for _ in range(3)}
        name = sorted(n for n in names if n)[0] if names else ""
        if name:
            return name
    # фоллбек без Faker
    nouns = ["Тех", "Софт", "Гео", "Диджитал", "Инфо", "Дата", "Сервис", "Про"]
    forms = ["Групп", "Системс", "Лаб", "Пром", "Солюшнс", "Холдинг", "Сервис"]
    return f"{rng.choice(nouns)}{rng.choice(forms)}"


def record_from_row(row: dict) -> dict:
    """
    Размечаем РОВНО под тренер:
      - PERSON  -> ФИО
      - ADDRESS -> адрес (в train_hf_ner.py станет LOC)
      - ORG     -> организация (из поля company|employer|organization|org,
                   либо синтезируется стабильно)
    Остальные поля НЕ размечаем, иначе тренер их отбросит.
    """
    parts: list[str] = []
    entities: list[list] = []

    fio = _1line(_first(row, "full_name", "fio", "name"))
    addr = _1line(_first(row, "address", "addr", "location"))
    # возможные ключи с организацией
    org = _1line(_first(row, "company", "employer", "organization", "org"))

    email = _first(row, "email", "mail")
    phone = _first(row, "phone", "tel", "telephone")
    birth = _first(row, "birth_date", "dob", "date_of_birth")
    passport = _first(row, "passport_id", "passport")
    bank = _first(row, "bank", "card", "card_number")

    # если явного org нет — синтезируем из e-mail или ФИО, чтобы класс ORG присутствовал
    if not org:
        org = _stable_company(email or fio)

    # ---- текст + индексы ----
    if fio:
        s, e = _add_field(parts, "ФИО: ", fio)
        entities.append([s, e, "PERSON"])
    else:
        _add_field(parts, "ФИО: ", "")

    _add_field(parts, "E-mail: ", email)
    _add_field(parts, "Телефон: ", phone)
    _add_field(parts, "Дата рождения: ", _1line(birth))

    if addr:
        s, e = _add_field(parts, "Адрес: ", addr)
        entities.append([s, e, "ADDRESS"])  # в тренере алиасится в LOC
    else:
        _add_field(parts, "Адрес: ", "")

    if org:
        s, e = _add_field(parts, "Организация: ", org)
        entities.append([s, e, "ORG"])
    else:
        _add_field(parts, "Организация: ", "")

    _add_field(parts, "Паспорт: ", passport)
    _add_field(parts, "Карта: ", bank)

    return {"text": "".join(parts), "entities": entities}


def make_jsonl(
    out_train: Path = LABELS_DIR / "train.jsonl",
    out_dev: Path = LABELS_DIR / "dev.jsonl",
    n_rows: int = 3000,
    ru_share: float = 0.5,
    dev_ratio: float = 2 / 6,
):
    df = make_df(n_rows=n_rows, ru_share=ru_share)
    try:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    except Exception:
        pass

    n_dev = max(1, int(len(df) * dev_ratio))
    dev_df = df.iloc[:n_dev]
    train_df = df.iloc[n_dev:]

    with (
        out_train.open("w", encoding="utf-8") as ftr,
        out_dev.open("w", encoding="utf-8") as fdv,
    ):
        for _, row in train_df.iterrows():
            rec = record_from_row(row if isinstance(row, dict) else row.to_dict())
            ftr.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for _, row in dev_df.iterrows():
            rec = record_from_row(row if isinstance(row, dict) else row.to_dict())
            fdv.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_df)} records to {out_train}")
    print(f"Wrote {len(dev_df)} records to {out_dev}")


if __name__ == "__main__":
    make_jsonl()

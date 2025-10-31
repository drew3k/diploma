from __future__ import annotations
from typing import Iterable
import regex as re  # unicode-friendly

# --- мягкие пробелы/невидимые символы ---
NBSP = "\u00a0"
CTRL0 = "\u200b\u200c\u200d\ufeff\u00ad"  # ZWSP/ZWNJ/ZWJ/BOM/soft hyphen
SOFT_SPACES = rf"\s{NBSP}"
SOFT_SEP = rf"[\s{NBSP}\-]"  # пробел/nbsp/дефис

# --- базовые регексы на входном тексте (включая NBSP/дефисы) ---
DEFAULT_REGEX = {
    # e-mail: whole local-part (letters/digits ._%+-) + @ + domain
    "EMAIL_ADDRESS": re.compile(
        r"(?<!\w)[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?!\w)"
    ),
    # RU phones: +7/8, допускаем NBSP и дефисы, (XXX)XXX-XX-XX и др.
    "PHONE_NUMBER": re.compile(
        rf"(?<!\d)(?:\+?7|8){SOFT_SEP}?(?:\(\d{{3}}\)|\d{{3}}){SOFT_SEP}?\d{{3}}{SOFT_SEP}?\d{{2}}{SOFT_SEP}?\d{{2}}(?!\d)"
    ),
    # Cards: 16 digits в виде 4-4-4-4; пробел/дефис/NBSP между группами
    "CREDIT_CARD": re.compile(rf"(?<!\d)(?:\d{{4}}(?:{SOFT_SEP})?){{3}}\d{{4}}(?!\d)"),
}

SUPPORTED_LABELS = {
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "IBAN_CODE",
    "US_SSN",
    "NRP",
}


class Span:
    def __init__(self, text: str, start: int, end: int, label: str):
        self.text = text
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self):
        return f"Span({self.label}@{self.start}:{self.end}='{self.text[:20]}…')"


def filter_labels(labels: Iterable[str] | None):
    if not labels:
        return SUPPORTED_LABELS
    return set(l for l in labels if l in SUPPORTED_LABELS)  # noqa: E741


# ---------- маскирование ----------
CYR_LETTER = re.compile(r"\p{IsCyrillic}", re.UNICODE)


def _mask_digits_preserve(text: str, keep_digits: int = 4) -> str:
    seen = 0
    out = []
    for ch in text:
        if ch.isdigit():
            out.append(ch if seen < keep_digits else "*")
            seen += 1
        else:
            out.append(ch)
    return "".join(out)


def mask_phone(text: str, keep_digits: int = 4) -> str:
    return _mask_digits_preserve(text, keep_digits=keep_digits)


def mask_card(text: str, keep_digits: int = 4) -> str:
    return _mask_digits_preserve(text, keep_digits=keep_digits)


def _mask_word_keep_first_letter(word: str) -> str:
    """Оставляем первую ВИДИМУЮ кириллическую букву, учитываем невидимые и дефисы."""
    parts = word.split("-")
    out_parts = []
    for part in parts:
        seen_first = False
        buf = []
        for ch in part:
            if ch in CTRL0:  # сохраняем, но не считаем первой
                buf.append(ch)
                continue
            if CYR_LETTER.fullmatch(ch):
                if not seen_first:
                    buf.append(ch)
                    seen_first = True
                else:
                    buf.append("*")
            else:
                buf.append(ch)
        out_parts.append("".join(buf))
    return "-".join(out_parts)


def mask_person(fullname: str) -> str:
    # токенизация по пробелам (в т.ч. NBSP), разделители сохраняем
    tokens = re.split(rf"([{SOFT_SPACES}]+)", fullname)
    for i in range(0, len(tokens), 2):
        if i < len(tokens) and tokens[i]:
            tokens[i] = _mask_word_keep_first_letter(tokens[i])
    return "".join(tokens)


def mask_email(email: str) -> str:
    """Маскируем локал-парт полностью, кроме первой буквы/цифры; точки/подчёркивания/плюсы сохраняем."""
    if "@" not in email:
        return "".join("*" if c.isalnum() else c for c in email)
    local, domain = email.split("@", 1)
    if not local:
        return "*" + ("@" + domain if domain else "")
    out_local = []
    kept = False
    for ch in local:
        if ch.isalnum():
            if not kept:
                out_local.append(ch)
                kept = True
            else:
                out_local.append("*")
        else:
            out_local.append(ch)
    return "".join(out_local) + "@" + domain


def mask_generic(text: str) -> str:
    return "".join("*" if c.isalnum() else c for c in text)


def smart_mask(label: str, text: str) -> str:
    if label == "PHONE_NUMBER":
        return mask_phone(text)
    if label == "CREDIT_CARD":
        return mask_card(text)
    if label == "PERSON":
        return mask_person(text)
    if label == "EMAIL_ADDRESS":
        return mask_email(text)
    # Защитная эвристика для "похожих на ФИО" случаев
    parts = re.findall(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*", text)
    if 1 <= len(parts) <= 3 and any(tok[0].isupper() for tok in parts):
        return mask_person(text)
    return mask_generic(text)

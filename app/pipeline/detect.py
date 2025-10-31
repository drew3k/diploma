from __future__ import annotations
from typing import List, Set, Dict, Optional

import regex as re

# Presidio (шаблоны: email/phone/cc/ip и т.п.)
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    PhoneRecognizer,
    CreditCardRecognizer,
    IpRecognizer,
)

# Наши утилиты
from .utils import Span, DEFAULT_REGEX, filter_labels
from app.settings import settings

# --- HF NER (опционально). Если модулей нет, просто отключаем шаг ---
try:
    from .hf_ner import HFNerRecognizer  # новый модуль из предыдущего шага

    _HFNER_AVAILABLE = True
except Exception:
    _HFNER_AVAILABLE = False

# --- Natasha + pymorphy2 (опционально). Если нет — шаг PERSON-RU будет пропущен. ---
try:
    from natasha import NamesExtractor, Doc
    import pymorphy2  # noqa

    _NATASHA_AVAILABLE = True
except Exception:
    _NATASHA_AVAILABLE = False


# -------------------------
# Константы/паттерны для русских ФИО (как бэкап/усиление к HF-NER)
# -------------------------
# Инициал(ы): А.А. / А. А.
INITIALS_RX = re.compile(r"(?:[А-ЯЁA-Z]\.){1,3}")
# Фамилия: заглавная + строчные, допускаем дефис
SURNAME_RX = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")
# «Грубая» форма ФИО: Фамилия Имя Отчество/инициалы (слегка либеральная)
FIO_RU_REGEX_REFINED = re.compile(
    rf"(?:{SURNAME_RX.pattern}\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+|"
    rf"\s+{INITIALS_RX.pattern})?"
    rf"|{INITIALS_RX.pattern}\s+{SURNAME_RX.pattern})"
)

# Список «анти-заголовков», чтобы не ловить разделы и меню
NEG_HEADERS = {
    "содержание",
    "оглавление",
    "приложение",
    "введение",
    "заключение",
    "библиография",
    "литература",
    "список литературы",
    "глоссарий",
    "определения",
    "термины",
    "примечания",
    "описание",
    "требования",
    "порядок",
    "договор",
    "договора",
    "статья",
    "статьи",
}

# Кэши для пресидио и HF-NER
_analyzers: Dict[str, AnalyzerEngine] = {}
_HF_RECOGNIZERS: Dict[str, HFNerRecognizer] = {}


# -------------------------
# Вспомогательные функции
# -------------------------
def _is_header_like(text: str) -> bool:
    t = text.strip().lower()
    return t in NEG_HEADERS or (len(t) < 48 and t.isupper())


def _extract_person_ru_natasha(text: str) -> List[Span]:
    """Бэкап: извлечение PERSON для RU через Natasha NamesExtractor."""
    if not _NATASHA_AVAILABLE:
        return []
    try:
        extractor = NamesExtractor()
        doc = Doc(text)
        doc.segment(segmenter=None)  # NamesExtractor не требует сегментатора
        spans: List[Span] = []
        for m in extractor(text):
            s, e = int(m.start), int(m.stop)
            frag = text[s:e]
            if _is_header_like(frag):
                continue
            spans.append(Span(frag, s, e, "PERSON"))
        return spans
    except Exception:
        return []


def build_analyzer(langs: list[str]) -> AnalyzerEngine:
    """Собирает Presidio AnalyzerEngine с нужными моделями spaCy."""
    lang_configs = []
    for l in langs:
        if l == "ru":
            ru_model_dir = settings.data_dir / "models" / "ru"
            nlp_name = str(ru_model_dir) if ru_model_dir.exists() else "ru_core_news_md"
            lang_configs.append({"lang_code": "ru", "model_name": nlp_name})
        elif l == "en":
            en_model_dir = settings.data_dir / "models" / "en"
            nlp_name = str(en_model_dir) if en_model_dir.exists() else "en_core_web_md"
            lang_configs.append({"lang_code": "en", "model_name": nlp_name})

    if not lang_configs:
        lang_configs = [{"lang_code": "ru", "model_name": "ru_core_news_md"}]

    provider = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": lang_configs}
    )
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=[c["lang_code"] for c in lang_configs],
    )

    analyzer.registry.add_recognizer(EmailRecognizer())
    analyzer.registry.add_recognizer(PhoneRecognizer())
    analyzer.registry.add_recognizer(CreditCardRecognizer())
    analyzer.registry.add_recognizer(IpRecognizer())

    return analyzer


# -------------------------
# Основная функция детекции
# -------------------------
def detect_spans(
    text: str, languages: list[str], limit_labels: Set[str] | None
) -> List[Span]:
    """
    Сводный детектор:
      1) Presidio (email/phone/cc/ip/…)
      2) Наши регэкспы (DEFAULT_REGEX)
      3) HF-NER (трансформер, с LoRA-адаптерами если есть)
      4) Наташа (PERSON, RU) как бэкап/усиление
      5) Слияние и дедупликация (приоритет длинных и критичных сущностей)
    """
    labels = filter_labels(limit_labels)
    spans: List[Span] = []

    if not text:
        return spans

    # нормализуем языки
    langs = [l for l in (languages or []) if l in ("ru", "en")]
    if not langs:
        langs = ["ru"]
    key = ",".join(sorted(langs))

    # 1) Presidio
    if key not in _analyzers:
        _analyzers[key] = build_analyzer(langs)
    analyzer = _analyzers[key]

    pres_results: List[RecognizerResult] = analyzer.analyze(
        text=text,
        language=langs[0],  # берем первый как основной
    )
    for e in pres_results:
        if e.entity_type in labels:
            spans.append(Span(text[e.start : e.end], e.start, e.end, e.entity_type))

    # 2) Регэкспы (наши дополнительные паттерны/варианты форматирования)
    for lbl, rx in DEFAULT_REGEX.items():
        if lbl in labels:
            for m in rx.finditer(text):
                spans.append(Span(m.group(0), m.start(), m.end(), lbl))

    # 3) HF-NER (PERSON/ADDRESS/ORG и т.п.) — если включено и доступно
    if settings.hf_ner_enabled and _HFNER_AVAILABLE:
        lang_for_ner = langs[0]
        recognizer = _HF_RECOGNIZERS.get(lang_for_ner)
        if recognizer is None:
            recognizer = HFNerRecognizer(lang=lang_for_ner)
            _HF_RECOGNIZERS[lang_for_ner] = recognizer

        try:
            spans.extend(recognizer.predict(text, limit_labels=set(labels)))
        except Exception:
            # при любых проблемах с HF-NER не валим весь пайплайн
            pass

    # 4) PERSON (RU) Natasha+pymorphy2 (бэкап/усиление)
    if "PERSON" in labels and ("ru" in langs):
        spans.extend(_extract_person_ru_natasha(text))
        # Плюс типовые шаблоны "Фамилия И.О." / "И.О. Фамилия"
        patt1 = re.compile(rf"({SURNAME_RX.pattern})\s*{INITIALS_RX.pattern}")
        patt2 = re.compile(rf"{INITIALS_RX.pattern}\s*({SURNAME_RX.pattern})")
        for pat in (patt1, patt2):
            for m in pat.finditer(text):
                spans.append(Span(m.group(0), m.start(), m.end(), "PERSON"))
        # «Грубая» ловля ФИО (последний шанс)
        for m in FIO_RU_REGEX_REFINED.finditer(text):
            frag = m.group(0)
            if not _is_header_like(frag):
                spans.append(Span(frag, m.start(), m.end(), "PERSON"))

    # 5) Слияние/приоритизация/дедупликация
    # Критичные типы всегда оставляем (правила имеют первенство):
    # EMAIL/PHONE/CREDIT_CARD/IP — даем им высокий приоритет.
    priority = {
        "EMAIL_ADDRESS": 5,
        "PHONE_NUMBER": 5,
        "CREDIT_CARD": 5,
        "IP_ADDRESS": 5,
        "PERSON": 3,
        "ADDRESS": 2,
        "ORG": 1,
    }
    spans.sort(key=lambda s: (s.start, -(s.end - s.start), -priority.get(s.label, 0)))

    pruned: List[Span] = []
    cur_end = -1
    for s in spans:
        if s.start >= cur_end:
            pruned.append(s)
            cur_end = s.end

    return pruned

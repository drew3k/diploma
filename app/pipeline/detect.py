from __future__ import annotations
from typing import List, Set, Dict

import regex as re

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    PhoneRecognizer,
    CreditCardRecognizer,
    IpRecognizer,
)

from .utils import Span, DEFAULT_REGEX, filter_labels
from app.settings import settings

try:
    from .hf_ner import HFNerRecognizer

    _HFNER_AVAILABLE = True
except Exception:
    _HFNER_AVAILABLE = False

try:
    from natasha import NamesExtractor, Doc

    _NATASHA_AVAILABLE = True
except Exception:
    _NATASHA_AVAILABLE = False


INITIALS_RX = re.compile(r"(?:[А-ЯЁA-Z]\.){1,3}")
SURNAME_RX = re.compile(r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?")
FIO_RU_REGEX_REFINED = re.compile(
    rf"(?:{SURNAME_RX.pattern}\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+|"
    rf"\s+{INITIALS_RX.pattern})?"
    rf"|{INITIALS_RX.pattern}\s+{SURNAME_RX.pattern})"
)

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


def _is_header_like(text: str) -> bool:
    t = text.strip().lower()
    return t in NEG_HEADERS or (len(t) < 48 and t.isupper())


_ADDR_LEFT_KEYS = (
    r"(?:г|город|р-н|район|пос|посёлок|поселок|дер|дп|с|село|пгт|ст|станица|мкр|микрорайон|"
    r"ул|улица|просп|пр-т|проспект|наб|набережная|пер|переулок|ш|шоссе|"
    r"пл|площадь|бул|бульвар|тракт|км|километр|владение|вл|строение|стр|дом|д|корп|к|лит|оф|офис|кв|квартира)"
)
_ADDR_LEFT_PATTERN = re.compile(
    rf"(?:^|[\s,])(?:{_ADDR_LEFT_KEYS})\.?(?:\s|$)", re.IGNORECASE
)
_ADDR_RIGHT_BOUNDARY = re.compile(r"[.;!?](?:\s+[\p{{Lu}}A-ZА-ЯЁ]|$)|\n")


def _expand_address_spans(text: str, spans: List[Span]) -> List[Span]:
    out: List[Span] = []
    for s in spans:
        if s.label not in ("ADDRESS", "LOCATION"):
            out.append(s)
            continue

        start = s.start
        left_slice_start = max(0, start - 32)
        left_slice = text[left_slice_start:start]
        m_left = list(_ADDR_LEFT_PATTERN.finditer(left_slice))
        if m_left:
            last = m_left[-1]
            start = left_slice_start + last.start()
            while start < s.start and text[start] in " ,\u00a0":
                start += 1

        end = s.end
        right_slice = text[end : end + 256]
        m_right = _ADDR_RIGHT_BOUNDARY.search(right_slice)
        if m_right:
            end = end + m_right.start()
        else:
            end = min(len(text), end + 256)

        while end > start and text[end - 1] in " ,\u00a0":
            end -= 1

        s.start, s.end, s.text = start, end, text[start:end]
        if s.label == "LOCATION":
            s.label = "ADDRESS"
        out.append(s)
    return out


_analyzers: Dict[str, AnalyzerEngine] = {}
_HF_RECOGNIZERS: Dict[str, HFNerRecognizer] = {}


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


def _extract_person_ru_natasha(text: str) -> List[Span]:
    if not _NATASHA_AVAILABLE:
        return []
    try:
        extractor = NamesExtractor()
        doc = Doc(text)
        spans: List[Span] = []
        for m in extractor(text):
            s0, e0 = int(m.start), int(m.stop)
            frag = text[s0:e0]
            if _is_header_like(frag):
                continue
            spans.append(Span(frag, s0, e0, "PERSON"))
        return spans
    except Exception:
        return []


def detect_spans(
    text: str, languages: list[str], limit_labels: Set[str] | None
) -> List[Span]:
    labels = filter_labels(limit_labels)
    spans: List[Span] = []

    if not text:
        return spans

    langs = [l for l in (languages or []) if l in ("ru", "en")]
    if not langs:
        langs = ["ru"]
    key = ",".join(sorted(langs))

    if key not in _analyzers:
        _analyzers[key] = build_analyzer(langs)
    analyzer = _analyzers[key]

    pres_results: List[RecognizerResult] = analyzer.analyze(
        text=text,
        language=langs[0],
    )
    for e in pres_results:
        if e.entity_type in labels:
            spans.append(Span(text[e.start : e.end], e.start, e.end, e.entity_type))

    for lbl, rx in DEFAULT_REGEX.items():
        if lbl in labels:
            for m in rx.finditer(text):
                spans.append(Span(m.group(0), m.start(), m.end(), lbl))

    if settings.hf_ner_enabled and _HFNER_AVAILABLE:
        lang_for_ner = langs[0]
        rec = _HF_RECOGNIZERS.get(lang_for_ner)
        if rec is None:
            rec = HFNerRecognizer(lang=lang_for_ner)
            _HF_RECOGNIZERS[lang_for_ner] = rec
        try:
            spans.extend(rec.predict(text, limit_labels=set(labels)))
        except Exception:
            pass

    if "PERSON" in labels and ("ru" in langs):
        spans.extend(_extract_person_ru_natasha(text))
        patt1 = re.compile(rf"({SURNAME_RX.pattern})\s*{INITIALS_RX.pattern}")
        patt2 = re.compile(rf"{INITIALS_RX.pattern}\s*({SURNAME_RX.pattern})")
        for pat in (patt1, patt2):
            for m in pat.finditer(text):
                spans.append(Span(m.group(0), m.start(), m.end(), "PERSON"))
        for m in FIO_RU_REGEX_REFINED.finditer(text):
            frag = m.group(0)
            if not _is_header_like(frag):
                spans.append(Span(frag, m.start(), m.end(), "PERSON"))

    spans = _expand_address_spans(text, spans)

    priority = {
        "EMAIL_ADDRESS": 5,
        "PHONE_NUMBER": 5,
        "CREDIT_CARD": 5,
        "IP_ADDRESS": 5,
        "PASSPORT_ID": 5,
        "PERSON": 3,
        "ADDRESS": 2,
        "ORGANIZATION": 1,
        "ORG": 1,
        "LOCATION": 2,
    }
    spans.sort(key=lambda s: (s.start, -(s.end - s.start), -priority.get(s.label, 0)))

    pruned: List[Span] = []
    cur_end = -1
    for s in spans:
        if s.start >= cur_end:
            pruned.append(s)
            cur_end = s.end

    return pruned

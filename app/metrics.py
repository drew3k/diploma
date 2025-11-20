from __future__ import annotations
from typing import Dict, List, Set, Tuple
import regex as re

from pipeline.utils import Span


def _to_key(s: Span) -> Tuple[int, int, str]:
    return (int(s.start), int(s.end), str(s.label))


def _match_counts(
    true_spans: List[Span], pred_spans: List[Span], labels: Set[str] | None = None
):
    """Exact-match по (start,end,label). Возвращает per-class TP/FP/FN и микро-итоги."""
    if labels is None:
        labels = set([s.label for s in true_spans] + [s.label for s in pred_spans])

    per: Dict[str, Dict[str, int]] = {
        lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in labels
    }

    true_by_lbl: Dict[str, Set[Tuple[int, int, str]]] = {lbl: set() for lbl in labels}
    pred_by_lbl: Dict[str, Set[Tuple[int, int, str]]] = {lbl: set() for lbl in labels}

    for s in true_spans:
        if s.label in labels:
            true_by_lbl[s.label].add(_to_key(s))
    for s in pred_spans:
        if s.label in labels:
            pred_by_lbl[s.label].add(_to_key(s))

    for lbl in labels:
        tp = len(true_by_lbl[lbl] & pred_by_lbl[lbl])
        fp = len(pred_by_lbl[lbl] - true_by_lbl[lbl])
        fn = len(true_by_lbl[lbl] - pred_by_lbl[lbl])
        per[lbl] = {"tp": tp, "fp": fp, "fn": fn}

    micro = {"tp": 0, "fp": 0, "fn": 0}
    for v in per.values():
        micro["tp"] += v["tp"]
        micro["fp"] += v["fp"]
        micro["fn"] += v["fn"]

    return per, micro


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def classification_metrics(
    true_spans: List[Span], pred_spans: List[Span], labels: Set[str] | None = None
):
    """Возвращает словарь с micro/macro и метриками по классам."""
    per_counts, micro_counts = _match_counts(true_spans, pred_spans, labels)

    per_scores: Dict[str, Dict[str, float]] = {}
    macro_accum = {"p": 0.0, "r": 0.0, "f1": 0.0}
    num_lbls = len(per_counts) if per_counts else 1
    for lbl, c in per_counts.items():
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        per_scores[lbl] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": c["tp"],
            "fp": c["fp"],
            "fn": c["fn"],
        }
        macro_accum["p"] += p
        macro_accum["r"] += r
        macro_accum["f1"] += f1

    micro_p, micro_r, micro_f1 = _prf(
        micro_counts["tp"], micro_counts["fp"], micro_counts["fn"]
    )
    macro = {k: (v / num_lbls) for k, v in macro_accum.items()}
    return {
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            **micro_counts,
        },
        "macro": macro,
        "per_class": per_scores,
    }


# -------- Метрики редактирования --------

_WS = r"[\s\u00A0]+"  # пробел/таб/NBSP


def _text_pattern(txt: str) -> re.Pattern:
    """Паттерн для поиска сырого текста ПД с терпимостью к пробелам."""
    esc = re.escape(txt)
    esc = esc.replace(r"\ ", _WS)
    return re.compile(esc)


def redaction_metrics(
    true_spans: List[Span], pred_spans: List[Span], redacted_text: str
):
    """
    Leakage: доля истинных ПД, которые остались видимыми после редактирования (по точному тексту).
    Over-redaction: FP/(TP+FP) — доля предсказанных сущностей, которых нет в эталоне.
    """
    # leakage
    leaked = 0
    for s in true_spans:
        if not s.text:
            continue
        rx = _text_pattern(s.text)
        if rx.search(redacted_text):
            leaked += 1
    total_true = len([s for s in true_spans if s.text])
    leakage = leaked / total_true if total_true else 0.0

    # over-redaction (по классификации)
    _, micro_counts = _match_counts(true_spans, pred_spans)
    tp, fp = micro_counts["tp"], micro_counts["fp"]
    over_redaction = fp / (tp + fp) if (tp + fp) else 0.0

    return {
        "pii_leakage": leakage,
        "leaked": leaked,
        "total_true": total_true,
        "over_redaction": over_redaction,
        "tp": tp,
        "fp": fp,
    }


def pretty_print(metrics: dict):
    """Красивая печать метрик в stdout."""

    def pct(x):
        return f"{x * 100:5.1f}%"

    micro = metrics["classification"]["micro"]
    macro = metrics["classification"]["macro"]

    print("\n=== Метрики классификации (детекция ПД) ===")
    print(
        f"Micro:  Precision={pct(micro['precision'])}  Recall={pct(micro['recall'])}  F1={pct(micro['f1'])}  "
        f"(TP={micro['tp']} FP={micro['fp']} FN={micro['fn']})"
    )
    print(
        f"Macro:  Precision={pct(macro['p'])}  Recall={pct(macro['r'])}  F1={pct(macro['f1'])}"
    )
    print("Per-class:")
    for lbl, s in metrics["classification"]["per_class"].items():
        print(
            f"  {lbl:14s}  P={pct(s['precision'])}  R={pct(s['recall'])}  F1={pct(s['f1'])}  "
            f"(TP={s['tp']} FP={s['fp']} FN={s['fn']})"
        )

    red = metrics["redaction"]
    print("\n=== Метрики редактирования ===")
    print(
        f"PII Leakage: {pct(red['pii_leakage'])}  (leaked={red['leaked']}/{red['total_true']})"
    )
    print(
        f"Over-redaction: {pct(red['over_redaction'])}  (TP={red['tp']} FP={red['fp']})"
    )

    if "timing_ms" in metrics:
        print("\n=== Производительность ===")
        print(f"Общее время: {metrics['timing_ms']} ms")

from __future__ import annotations
from typing import Dict, List, Set, Tuple
import regex as re

from pipeline.utils import Span


def _to_key(s: Span) -> Tuple[int, int, str]:
    return (int(s.start), int(s.end), str(s.label))


def _match_counts(
    true_spans: List[Span], pred_spans: List[Span], labels: Set[str] | None = None
):
    """Exact-match по (start,end,label). Считаем per-class TP/FP/FN и микросуммы."""
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
    """Считаем precision/recall/F1 и accuracy (micro/macro + per-class)."""
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
    denom_micro = micro_counts["tp"] + micro_counts["fp"] + micro_counts["fn"]
    micro_acc = micro_counts["tp"] / denom_micro if denom_micro else 0.0

    macro = {k: (v / num_lbls) for k, v in macro_accum.items()}
    return {
        "micro": {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "accuracy": micro_acc,
            **micro_counts,
        },
        "macro": macro,
        "per_class": per_scores,
    }


# --------- вспомогательные вещи для старых метрик редактирования (пока оставим, но не печатаем) ---------

_WS = r"[\s\u00A0]+"


def _text_pattern(txt: str) -> re.Pattern:
    esc = re.escape(txt)
    esc = esc.replace(r"\ ", _WS)
    return re.compile(esc)


def redaction_metrics(
    true_spans: List[Span], pred_spans: List[Span], redacted_text: str
):
    """Старая метрика утечек/over-redaction; сейчас в pretty_print не используется."""
    leaked = 0
    for s in true_spans:
        if not s.text:
            continue
        rx = _text_pattern(s.text)
        if rx.search(redacted_text):
            leaked += 1
    total_true = len([s for s in true_spans if s.text])
    leakage = leaked / total_true if total_true else 0.0

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
    """Печать удобочитаемых метрик качества в stdout."""

    def pct(x: float) -> str:
        return f"{x * 100:5.1f}%"

    micro = metrics["micro"]
    macro = metrics["macro"]
    per_class = metrics["per_class"]

    # Micro accuracy (если вдруг не посчитана)
    micro_acc = micro.get("accuracy")
    if micro_acc is None:
        denom = micro.get("tp", 0) + micro.get("fp", 0) + micro.get("fn", 0)
        micro_acc = micro.get("tp", 0) / denom if denom else 0.0

    # Macro accuracy: среднее Accuracy по классам
    total_acc = 0.0
    n_labels = 0
    for s in per_class.values():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        denom = tp + fp + fn
        acc_i = tp / denom if denom else 0.0
        total_acc += acc_i
        n_labels += 1
    macro_acc = total_acc / n_labels if n_labels else 0.0

    print("\n=== Качество распознавания сущностей (exact match) ===")
    print(
        f"Micro:  Accuracy={pct(micro_acc)}  "
        f"Precision={pct(micro['precision'])}  "
        f"Recall={pct(micro['recall'])}  "
        f"F1-score={pct(micro['f1'])}"
    )
    print(
        f"Macro:  Accuracy={pct(macro_acc)}  "
        f"Precision={pct(macro['p'])}  "
        f"Recall={pct(macro['r'])}  "
        f"F1-score={pct(macro['f1'])}"
    )

    print("\nPer-class:")
    print(
        "  {:14s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
            "Label", "Accuracy", "Precision", "Recall", "F1-score"
        )
    )
    for lbl, s in per_class.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        denom = tp + fp + fn
        acc = tp / denom if denom else 0.0
        print(
            f"  {lbl:14s}  {pct(acc):>10s}  "
            f"{pct(s['precision']):>10s}  "
            f"{pct(s['recall']):>10s}  "
            f"{pct(s['f1']):>10s}"
        )

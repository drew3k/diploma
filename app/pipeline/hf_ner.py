from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import inspect
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

try:
    from peft import PeftModel

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

from settings import settings
from .utils import Span

HF_NER_BIO_LABELS = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


@dataclass
class HFNerConfig:
    model_name: str
    threshold: float = 0.80
    label_map: dict[str, str] | None = None
    max_length: int = 512
    adapters_dir: Optional[Path] = None  # где лежат LoRA/адаптеры


class HFNerRecognizer:
    def __init__(self, lang: str = "ru", cfg: HFNerConfig | None = None):
        if cfg is None:
            adapters_dir = settings.hf_ner_adapters_dir or (
                settings.data_dir / "hf_ner"
            )
            self.cfg = HFNerConfig(
                model_name=settings.hf_ner_model_name,
                threshold=settings.hf_ner_threshold,
                label_map=settings.hf_ner_label_map,
                max_length=settings.hf_ner_max_length,
                adapters_dir=(adapters_dir / lang),
            )
        else:
            self.cfg = cfg

        self.lang = lang
        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

        num_labels = len(HF_NER_BIO_LABELS)
        base_model = AutoModelForTokenClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=num_labels,
        )
        base_model.config.id2label = {i: lab for i, lab in enumerate(HF_NER_BIO_LABELS)}
        base_model.config.label2id = {lab: i for i, lab in enumerate(HF_NER_BIO_LABELS)}

        if self.cfg.adapters_dir and self.cfg.adapters_dir.exists() and _HAS_PEFT:
            try:
                base_model = PeftModel.from_pretrained(
                    base_model, str(self.cfg.adapters_dir)
                )
                print(f"[hf_ner] Loaded adapters from {self.cfg.adapters_dir}")
            except Exception as e:
                print(f"[hf_ner] Failed to load adapters: {e}")

        self.pipe = pipeline(
            "token-classification",
            model=base_model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=self.device,
        )

        self.label_map = self.cfg.label_map or {}

    def predict(self, text: str, limit_labels: set[str] | None = None) -> List[Span]:
        if not text:
            return []

        pipe_kwargs = {}
        try:
            sig = inspect.signature(self.pipe._sanitize_parameters)
            params = sig.parameters
            if "truncation" in params:
                pipe_kwargs["truncation"] = True
            if "max_length" in params:
                pipe_kwargs["max_length"] = self.cfg.max_length
            if "stride" in params:
                pipe_kwargs["stride"] = 128
            if "return_all_scores" in params:
                pipe_kwargs["return_all_scores"] = False
        except Exception:
            pipe_kwargs = {}

        results = self.pipe(text, **pipe_kwargs)

        spans: list[Span] = []
        for r in results:
            raw = r.get("entity_group") or r.get("entity") or ""
            conf = float(r.get("score", 0.0))
            start = int(r.get("start", 0))
            end = int(r.get("end", 0))
            if end <= start:
                continue

            mapped = self.label_map.get(raw, raw)
            if limit_labels and mapped not in limit_labels:
                continue
            if conf < self.cfg.threshold:
                continue

            frag = text[start:end]
            spans.append(Span(frag, start, end, mapped))

        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        ded: list[Span] = []
        cur_end = -1
        for s in spans:
            if s.start >= cur_end:
                ded.append(s)
                cur_end = s.end
        return ded

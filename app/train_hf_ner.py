# train_hf_ner.py  — обучение LoRA-адаптера для NER (PERSON/LOC/ORG)
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Tuple
import inspect

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from settings import settings  # лежит рядом с main.py


@dataclass
class Rec:
    text: str
    entities: List[Tuple[int, int, str]]  # (start, end, label)


# нормализация ярлыков из JSONL
_LABEL_ALIASES = {
    "PERSON": "PER",
    "PER": "PER",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "ADDRESS": "LOC",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
}


def load_jsonl(path: Path) -> List[Rec]:
    out: List[Rec] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        ents = []
        for s, e, lbl in obj.get("entities", []):
            lbl_norm = _LABEL_ALIASES.get(str(lbl).upper().strip())
            if not lbl_norm:
                continue
            ents.append((int(s), int(e), lbl_norm))
        out.append(Rec(text=obj["text"], entities=ents))
    return out


def to_bio_encodings(
    tokenizer, recs: List[Rec], label2id: dict[str, int], max_length: int = 512
):
    encoded = []
    for r in recs:
        text = r.text
        e = tokenizer(
            text, return_offsets_mapping=True, truncation=True, max_length=max_length
        )
        offsets = e.pop("offset_mapping")
        labels = [-100] * len(offsets)

        for s, e_, lbl in r.entities:
            begin_done = False
            for i, (cs, ce) in enumerate(offsets):
                if cs is None or ce is None or ce <= cs:
                    continue
                if ce <= s or cs >= e_:
                    continue
                tag = ("B-" if not begin_done else "I-") + lbl
                begin_done = True
                labels[i] = label2id.get(tag, -100)

        e["labels"] = labels
        encoded.append(e)

    keys = encoded[0].keys()
    data = {k: [ex[k] for ex in encoded] for k in keys}
    return data


def main(
    train_path: str = "data/labels/train.jsonl",
    dev_path: str = "data/labels/dev.jsonl",
    base_model: str = settings.hf_ner_model_name,
    out_dir: str = "data/hf_ner/ru",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    epochs: int = 3,
    lr: float = 3e-5,
    batch: int = 8,
    max_length: int = 512,
):
    train_recs = load_jsonl(Path(train_path))
    dev_recs = load_jsonl(Path(dev_path))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=6,  # BIO: PER/ORG/LOC -> 6 меток
        problem_type="token_classification",
    )

    labels = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for k, i in label2id.items()}
    base.config.label2id = label2id
    base.config.id2label = id2label

    train_data = to_bio_encodings(
        tokenizer, train_recs, label2id, max_length=max_length
    )
    dev_data = to_bio_encodings(tokenizer, dev_recs, label2id, max_length=max_length)
    ds_train = Dataset.from_dict(train_data)
    ds_dev = Dataset.from_dict(dev_data)

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value", "key", "dense", "output.dense"],
        bias="none",
        task_type="TOKEN_CLS",
    )
    model = get_peft_model(base, lora_cfg)

    collator = DataCollatorForTokenClassification(tokenizer)

    # --- Совместимые с разными версиями transformers аргументы ---
    sig = inspect.signature(TrainingArguments.__init__)

    def supports(arg: str) -> bool:
        return arg in sig.parameters

    ta_kwargs = {
        "output_dir": str(out_path),
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": batch,
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "logging_steps": 50,
    }
    # опциональные по версии
    if supports("evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = "epoch"
    if supports("save_strategy"):
        ta_kwargs["save_strategy"] = "epoch"
    if supports("fp16"):
        ta_kwargs["fp16"] = torch.cuda.is_available()
    if supports("report_to"):
        ta_kwargs["report_to"] = []  # отключить отчёты (wandb и т.п.)
    if supports("save_total_limit"):
        ta_kwargs["save_total_limit"] = 2

    args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    model.save_pretrained(str(out_path))
    print(f"[train_hf_ner] Saved LoRA adapters to {out_path.resolve()}")


if __name__ == "__main__":
    main()

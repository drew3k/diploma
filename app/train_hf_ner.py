from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

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

from app.settings import settings


@dataclass
class Rec:
    text: str
    entities: list[tuple[int, int, str]]


def load_jsonl(path: Path) -> list[Rec]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        ents = [(int(s), int(e), str(lbl)) for s, e, lbl in obj.get("entities", [])]
        out.append(Rec(text=obj["text"], entities=ents))
    return out


def char_to_token_labels(
    tokenizer, text: str, entities: list[tuple[int, int, str]], label2id: dict[str, int]
):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
    labels = [-100] * len(enc["offset_mapping"])
    for s, e, lbl in entities:
        for i, (cs, ce) in enumerate(enc["offset_mapping"]):
            if cs is None:  # fast tokenizers sometimes
                continue
            if ce <= cs:
                continue
            # простая BIO схема
            if cs >= e or ce <= s:
                continue
            tag = "B-" + lbl if labels[i] == -100 else "I-" + lbl
            labels[i] = label2id.get(tag, -100)
    enc.pop("offset_mapping")
    enc["labels"] = labels
    return enc


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
):
    train_recs = load_jsonl(Path(train_path))
    dev_recs = load_jsonl(Path(dev_path))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=6, problem_type="token_classification"
    )

    # фиксируем нашу схему меток
    labels = ["B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for k, i in label2id.items()}
    base.config.label2id = label2id
    base.config.id2label = id2label

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "query",
            "value",
            "key",
            "dense",
            "output.dense",
        ],  # общие имена для BERT-подобных
        bias="none",
        task_type="TOKEN_CLS",
    )
    model = get_peft_model(base, lora_cfg)

    def _enc(r: Rec):
        return char_to_token_labels(tokenizer, r.text, r.entities, label2id)

    ds_train = Dataset.from_list(
        [{"text": r.text, "entities": r.entities} for r in train_recs]
    ).map(_enc, batched=False)
    ds_dev = Dataset.from_list(
        [{"text": r.text, "entities": r.entities} for r in dev_recs]
    ).map(_enc, batched=False)

    collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_dev,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    print(f"Saved LoRA adapters to {out_dir}")


if __name__ == "__main__":
    main()

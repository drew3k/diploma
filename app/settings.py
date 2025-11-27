from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("./data"))
    max_file_mb: int = 50

    # --- HF NER ---
    hf_ner_enabled: bool = True
    hf_ner_model_name: str = "DeepPavlov/rubert-base-cased"
    hf_ner_label_map: dict[str, str] = {
        "B-PER": "PERSON",
        "I-PER": "PERSON",
        "PER": "PERSON",
        "B-LOC": "ADDRESS",
        "I-LOC": "ADDRESS",
        "LOC": "ADDRESS",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "ORG": "ORG",
    }
    hf_ner_threshold: float = 0.7
    hf_ner_max_length: int = 512
    # Путь к каталогу с LoRA-адаптерами; по умолчанию data/hf_ner/<lang>
    hf_ner_adapters_dir: Path | None = None  # None -> data/hf_ner

    class Config:
        env_file = ".env"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "in").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "out").mkdir(parents=True, exist_ok=True)

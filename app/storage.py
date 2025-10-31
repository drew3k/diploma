from pathlib import Path
from .settings import settings
import uuid

IN = settings.data_dir / "in"
OUT = settings.data_dir / "out"


def save_upload(filename: str, content: bytes) -> Path:
    stem = f"{uuid.uuid4().hex}__{Path(filename).name}"
    path = IN / stem
    path.write_bytes(content)
    return path


def out_path_for(inp: Path, suffix: str) -> Path:
    return OUT / f"{inp.stem}.redacted{suffix}"


def public_url(p: Path) -> str:
    return f"/api/file/{p.name}"

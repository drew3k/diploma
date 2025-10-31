from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.responses import StreamingResponse
from typing import List
from pathlib import Path
from io import BytesIO
from secrets import token_urlsafe
from urllib.parse import quote
import unicodedata
from app.pipeline.feedback import log_candidates
from app.models import ProcessResult
from app.settings import settings
from app.storage import save_upload, out_path_for, public_url, OUT
from app.pipeline.detect import detect_spans
from app.pipeline.utils import DEFAULT_REGEX
from app.pipeline.redact_pdf import redact_pdf
from app.pipeline.cleanse_docx import cleanse_docx

app = FastAPI(title="PD Redactor Service", version="0.2.2")

# --------------------------- Стартовая страница ----------------------------
INDEX = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX


@app.get("/api/health")
def health():
    return {"status": "ok"}


# --------------------------- Классический API (сохранение на диск) --------
@app.post("/api/process", response_model=List[ProcessResult])
async def process_files(
    files: List[UploadFile] = File(...),
    policy: str = Form("mask"),
    languages: str = Form("ru,en"),
    types: str | None = Form(None),
):
    if policy not in ("mask", "remove"):
        raise HTTPException(400, "policy must be 'mask' or 'remove'")

    results: List[ProcessResult] = []

    for uf in files:
        content = await uf.read()
        if len(content) > settings.max_file_mb * 1024 * 1024:
            raise HTTPException(413, f"{uf.filename}: file too large")

        inp = save_upload(uf.filename, content)
        suffix = Path(uf.filename).suffix.lower()

        if suffix == ".pdf":
            import fitz

            with fitz.open(inp) as doc:
                text_for_detection = "\n".join(page.get_text() for page in doc)
        elif suffix == ".docx":
            from docx import Document

            d = Document(str(inp))
            # Base text from paragraphs
            text_for_detection = "\n".join(p.text for p in d.paragraphs)

            # Also include emails from hyperlink targets (e.g., mailto: links)
            try:
                EMAIL_RX = DEFAULT_REGEX.get("EMAIL_ADDRESS")
                if EMAIL_RX is not None:

                    def _emails_from_part(part) -> list[str]:
                        out: list[str] = []
                        try:
                            for rel in part.rels.values():
                                # Word hyperlink relationships
                                if (
                                    rel.reltype
                                    == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
                                ):
                                    target = getattr(rel, "target_ref", "") or ""
                                    m = EMAIL_RX.search(target)
                                    if m:
                                        out.append(m.group(0))
                        except Exception:
                            pass
                        return out

                    emails: set[str] = set(_emails_from_part(d.part))
                    # headers/footers have separate parts and rels
                    for section in d.sections:
                        if getattr(section, "header", None):
                            emails.update(_emails_from_part(section.header.part))
                        if getattr(section, "footer", None):
                            emails.update(_emails_from_part(section.footer.part))
                    if emails:
                        text_for_detection += "\n" + "\n".join(sorted(emails))
            except Exception:
                # Best-effort: ignore hyperlink extraction failures
                pass
        else:
            raise HTTPException(415, f"Unsupported type: {suffix}")

        spans = detect_spans(
            text_for_detection,
            languages.split(","),
            set(types.split(",")) if types else None,
        )

        log_candidates(text_for_detection, spans, source="api")

        if suffix == ".pdf":
            out = out_path_for(inp, ".pdf")
            found = redact_pdf(inp, out, spans, policy)
            results.append(
                ProcessResult(
                    input_name=uf.filename,
                    output_name=out.name,
                    output_url=public_url(out),
                    found=found,
                    filetype="pdf",
                )
            )
        else:
            out = out_path_for(inp, ".docx")
            found = cleanse_docx(inp, out, spans, policy)
            results.append(
                ProcessResult(
                    input_name=uf.filename,
                    output_name=out.name,
                    output_url=public_url(out),
                    found=found,
                    filetype="docx",
                )
            )

    return results


@app.get("/api/file/{name}")
def download_saved(name: str):
    p = OUT / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# --------------------------- Обработка в памяти + красивый скачиватор -----

# Простая in-memory кэш-таблица: token -> {"bytes":..., "media":..., "filename":...}
DOWNLOAD_CACHE: dict[str, dict[str, bytes | str]] = {}


def _detect_text_from_pdf_bytes(data: bytes) -> str:
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _redact_pdf_in_memory(data: bytes, spans, policy: str) -> bytes:
    import fitz
    from app.pipeline.utils import smart_mask

    BLACK = (0, 0, 0)
    WHITE = (1, 1, 1)

    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            for s in spans:
                txt = (s.text or "").strip()
                if not txt:
                    continue
                rects = page.search_for(txt, quads=False)
                if not rects:
                    continue
                overlay_text = smart_mask(s.label, s.text) if policy == "mask" else None
                fill_color = WHITE if policy == "mask" else BLACK
                for r in rects:
                    page.add_redact_annot(r, fill=fill_color, text=overlay_text)
            page.apply_redactions(images=(policy == "remove"))

        try:
            doc.set_metadata({})
        except Exception:
            pass

        return doc.tobytes(garbage=4, deflate=True, clean=True)


def _detect_text_from_docx_bytes(data: bytes) -> str:
    from docx import Document

    d = Document(BytesIO(data))
    # Base text from paragraphs
    text = "\n".join(p.text for p in d.paragraphs)

    # Also include emails from hyperlink targets (e.g., mailto: links)
    try:
        EMAIL_RX = DEFAULT_REGEX.get("EMAIL_ADDRESS")
        if EMAIL_RX is not None:

            def _emails_from_part(part) -> list[str]:
                out: list[str] = []
                try:
                    for rel in part.rels.values():
                        if (
                            rel.reltype
                            == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
                        ):
                            target = getattr(rel, "target_ref", "") or ""
                            m = EMAIL_RX.search(target)
                            if m:
                                out.append(m.group(0))
                except Exception:
                    pass
                return out

            emails: set[str] = set(_emails_from_part(d.part))
            for section in d.sections:
                if getattr(section, "header", None):
                    emails.update(_emails_from_part(section.header.part))
                if getattr(section, "footer", None):
                    emails.update(_emails_from_part(section.footer.part))
            if emails:
                text += "\n" + "\n".join(sorted(emails))
    except Exception:
        pass

    return text


def _cleanse_docx_in_memory(data: bytes, spans, policy: str) -> bytes:
    from docx import Document
    from app.pipeline.cleanse_docx import (
        _replace_in_headers_footers,
        _replace_in_paragraphs,
        _replace_in_tables,
        _clear_core_properties,
        _strip_comments_and_tracked_changes,
        _sanitize_hyperlinks_with_emails,
    )

    doc = Document(BytesIO(data))
    _strip_comments_and_tracked_changes(doc)
    _replace_in_headers_footers(doc, spans, policy)
    _replace_in_paragraphs(doc, spans, policy)
    _replace_in_tables(doc, spans, policy)
    _sanitize_hyperlinks_with_emails(doc, policy)
    _clear_core_properties(doc)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


@app.post("/web/submit")
async def web_submit(
    file: UploadFile = File(...),
    policy: str = Form("mask"),
    languages: str = Form("ru,en"),
    types: str | None = Form(None),
):
    if policy not in ("mask", "remove"):
        raise HTTPException(400, "policy must be 'mask' or 'remove'")

    content = await file.read()
    suffix = Path(file.filename).suffix.lower()

    if suffix == ".pdf":
        text_for_detection = _detect_text_from_pdf_bytes(content)
    elif suffix == ".docx":
        text_for_detection = _detect_text_from_docx_bytes(content)
    else:
        raise HTTPException(415, f"Unsupported type: {suffix}")

    spans = detect_spans(
        text_for_detection,
        languages.split(","),
        set(types.split(",")) if types else None,
    )

    log_candidates(text_for_detection, spans, source="web")

    if suffix == ".pdf":
        processed = _redact_pdf_in_memory(content, spans, policy)
        media = "application/pdf"
    else:
        processed = _cleanse_docx_in_memory(content, spans, policy)
        media = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    token = token_urlsafe(16)
    download_name = f"{Path(file.filename).stem}_redacted{suffix}"
    DOWNLOAD_CACHE[token] = {
        "bytes": processed,
        "media": media,
        "filename": download_name,
    }

    return RedirectResponse(url=f"/download/{token}", status_code=303)


@app.get("/download/{token}", response_class=HTMLResponse)
def download_page(token: str):
    item = DOWNLOAD_CACHE.get(token)
    if not item:
        return HTMLResponse(
            "<h2 style='font-family:Segoe UI,Roboto,Arial'>Ссылка устарела или не найдена.</h2>",
            status_code=404,
        )

    filename = item["filename"]
    html = f"""
<!doctype html>
<meta charset="utf-8"/>
<title>Скачать файл</title>
<style>
body{{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial;background:#111;margin:0;color:#eee}}
.header{{text-align:center;padding:60px 20px}}
.header h1{{font-size:42px;margin:0 0 8px}}
.header p{{font-size:18px;color:#bbb}}
.container{{display:flex;justify-content:center}}
.btn{{display:inline-flex;align-items:center;gap:12px;padding:16px 28px;font-size:20px;
     background:#e02424;color:#fff;border:none;border-radius:14px;text-decoration:none;box-shadow:0 6px 20px rgba(224,36,36,.25)}}
.btn:hover{{filter:brightness(1.05)}}
.card{{background:#1b1b1f;border-radius:16px;box-shadow:0 6px 24px rgba(0,0,0,.24);padding:32px;max-width:680px;margin:24px auto;text-align:center}}
.file{{font-weight:600;color:#fff}}
</style>
<div class="header">
  <h1>Файл готов</h1>
  <p>Результат обработки: <span class="file">{filename}</span></p>
</div>
<div class="container">
  <a class="btn" href="/api/download/{token}">
    &#128229; Скачать файл
  </a>
</div>
<div class="card">
  <p>Нажми «Скачать файл». Браузер откроет диалог выбора папки для сохранения.</p>
</div>
"""
    return HTMLResponse(html)


@app.get("/api/download/{token}")
async def api_download(token: str, request: Request):
    """
    Отдаём файл из памяти как attachment.
    Фикс для не-ASCII имён: используем filename* (RFC 5987) + ASCII-фолбэк.
    """
    item = DOWNLOAD_CACHE.get(token)
    if not item:
        raise HTTPException(404, "Файл не найден или ссылка устарела.")

    data: bytes = item["bytes"]  # байты файла
    media: str = item["media"]  # MIME типа "application/pdf" или DOCX
    filename: str = item[
        "filename"
    ]  # имя *.redacted.pdf|docx (может быть на кириллице)

    # ASCII-фолбэк (на случай кириллицы): убираем диакритику/не-ascii
    ascii_fallback = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    if not ascii_fallback:
        ascii_fallback = "download" + Path(filename).suffix

    # RFC 5987: filename* с URL-экранированием UTF-8
    cd = (
        f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{quote(filename)}"
    )

    headers = {
        "Content-Disposition": cd,
        "Cache-Control": "no-store",
    }

    return StreamingResponse(BytesIO(data), media_type=media, headers=headers)


# --------------------------- Точка входа -----------------------------------
@app.post("/api/train")
async def api_train(
    rows: int = Form(1000),
    policy: str = Form("mask"),
    languages: str = Form("ru,en"),
):
    from app.pipeline.train import train_from_fake

    langs = [l for l in languages.split(",") if l]  # noqa: E741
    metrics = train_from_fake(n_rows=rows, policy=policy, languages=langs)
    return metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

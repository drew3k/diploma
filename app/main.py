from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

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
    files: List[UploadFile] = File(...),
    policy: str = Form("mask"),
    languages: str = Form("ru,en"),
    types: str | None = Form(None),
):
    if policy not in ("mask", "remove"):
        raise HTTPException(400, "policy must be 'mask' or 'remove'")

    processed_items: List[dict[str, str]] = []

    for file in files:
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
        processed_items.append({"token": token, "filename": download_name})

    if len(processed_items) == 1:
        return RedirectResponse(url=f"/download/{processed_items[0]['token']}", status_code=303)

    links = "".join(
    f"""
    <a class="file-link" href="/api/download/{item["token"]}">
        <span class="file-icon">📄</span>
        <span class="file-info">
            <strong>{item["filename"]}</strong>
            <small>Скачать обезличенный документ</small>
        </span>
        <span class="download-icon">⬇</span>
    </a>
    """
    for item in processed_items
    )
    html = f"""
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Файлы готовы — PD Redactor</title>

    <link rel="stylesheet" href="/static/result-list.css"/>
</head>

<body>
    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>

    <main class="page">
        <header class="topbar">
            <div class="brand">
                <div class="brand-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 3L20 7V12C20 17 16.5 20.5 12 21C7.5 20.5 4 17 4 12V7L12 3Z" stroke="white" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M9 12L11 14L15.5 9.5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <div>PD Redactor</div>
                    <small class="brand-subtitle">Secure document anonymization</small>
                </div>
            </div>

            <div class="status-pill">
                <span class="pulse"></span>
                Обработка завершена
            </div>
        </header>

        <section class="result-wrap">
            <div class="panel">
                <div class="panel-inner">
                    <div class="success-icon">
                        <svg width="44" height="44" viewBox="0 0 24 24" fill="none">
                            <path d="M20 6L9 17L4 12" stroke="white" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>

                    <h1>
                        Файлы <span class="gradient-text">готовы</span>
                    </h1>

                    <p class="subtitle">
                        Все загруженные документы успешно обработаны. Скачайте обезличенные версии файлов по ссылкам ниже.
                    </p>

                    <div class="files-list">
                        {links}
                    </div>

                    <div class="actions">
                        <a class="btn-secondary" href="/">Обработать ещё документы</a>
                    </div>
                </div>
            </div>
        </section>
    </main>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/download/{token}", response_class=HTMLResponse)
def download_page(token: str):
    item = DOWNLOAD_CACHE.get(token)
    if not item:
        html = """
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Ссылка не найдена</title>
    <link rel="stylesheet" href="/static/download-missing.css"/>
</head>
<body>
    <div class="card">
        <h1>Ссылка устарела</h1>
        <p>Файл не найден в кэше сервера. Загрузите документ повторно и запустите обработку ещё раз.</p>
        <a href="/">Вернуться на главную</a>
    </div>
</body>
</html>
"""
        return HTMLResponse(html, status_code=404)

    filename = item["filename"]

    html = f"""
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Файл готов — PD Redactor</title>

    <link rel="stylesheet" href="/static/download.css"/>
</head>

<body>
    <div class="orb one"></div>
    <div class="orb two"></div>
    <div class="orb three"></div>

    <main class="page">
        <header class="topbar">
            <div class="brand">
                <div class="brand-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 3L20 7V12C20 17 16.5 20.5 12 21C7.5 20.5 4 17 4 12V7L12 3Z" stroke="white" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M9 12L11 14L15.5 9.5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div>
                    <div>PD Redactor</div>
                    <small class="brand-subtitle">Secure document anonymization</small>
                </div>
            </div>

            <div class="status-pill">
                <span class="pulse"></span>
                Обработка завершена
            </div>
        </header>

        <section class="result-wrap">
            <div class="panel">
                <div class="panel-inner">
                    <div class="success-icon">
                        <svg width="44" height="44" viewBox="0 0 24 24" fill="none">
                            <path d="M20 6L9 17L4 12" stroke="white" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>
                        </svg>
                    </div>

                    <h1>
                        Файл
                        <span class="gradient-text">готов</span>
                    </h1>

                    <p class="subtitle">
                        Обезличенная версия документа успешно сформирована. Теперь файл можно скачать
                        и использовать для безопасной передачи или хранения.
                    </p>

                    <div class="file-card">
                        <span>📄</span>
                        <span class="file-name">{filename}</span>
                    </div>

                    <div class="actions">
                        <a class="btn" href="/api/download/{token}">
                            <span>⬇</span>
                            Скачать файл
                        </a>

                        <a class="btn-secondary" href="/">
                            Обработать ещё
                        </a>
                    </div>

                    <div class="steps">
                        <div class="step">
                            <strong>1. Detection</strong>
                            <span>Персональные данные были обнаружены системой.</span>
                        </div>

                        <div class="step">
                            <strong>2. Redaction</strong>
                            <span>Фрагменты были скрыты согласно выбранной политике.</span>
                        </div>

                        <div class="step">
                            <strong>3. Delivery</strong>
                            <span>Готовый документ доступен для скачивания.</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    </main>
</body>
</html>
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

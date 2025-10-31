from __future__ import annotations
from pathlib import Path
from typing import List

import fitz

from .utils import Span, smart_mask


BLACK = (0, 0, 0)
WHITE = (1, 1, 1)


def _overlay_text_for_span(span: Span, policy: str) -> str | None:
    if policy == "mask":
        try:
            return smart_mask(span.label, span.text)
        except Exception:
            return ""
    return None


def redact_pdf(inp: Path, out: Path, spans: List[Span], policy: str = "mask") -> int:
    """Perform true redaction on the PDF.

    - Removes matched content via redact annotations + apply_redactions.
    - If policy == "mask", overlays masked replacement text inside the redaction box.
    - Clears document metadata before saving.
    """
    if not spans:
        out.write_bytes(inp.read_bytes())
        return 0

    doc = fitz.open(inp)
    total_rects = 0

    try:
        for page in doc:
            for s in spans:
                txt = (s.text or "").strip()
                if not txt:
                    continue
                rects = page.search_for(txt, quads=False)
                if not rects:
                    continue
                overlay_text = _overlay_text_for_span(s, policy)
                fill_color = WHITE if policy == "mask" else BLACK
                for r in rects:
                    total_rects += 1
                    page.add_redact_annot(
                        r,
                        fill=fill_color,
                        text=overlay_text,
                    )
            page.apply_redactions(images=(policy == "remove"))

        try:
            doc.set_metadata({})
        except Exception:
            pass

        doc.save(out, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    return total_rects

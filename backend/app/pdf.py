import re

import pymupdf

from .config import OCR_ENABLED, TESSDATA_DIR


def extract_external_links(page) -> list[str]:
    """Return visible labels and targets from external PDF link annotations."""
    links = []
    seen = set()
    for link in page.get_links():
        uri = (link.get("uri") or "").strip()
        if not uri or not uri.lower().startswith(("http://", "https://", "mailto:")):
            continue
        if uri in seen:
            continue
        seen.add(uri)
        label = ""
        rectangle = link.get("from")
        if rectangle:
            label = re.sub(r"\s+", " ", page.get_textbox(rectangle)).strip(" :-\n")
        links.append(f"{label}: {uri}" if label and label != uri else uri)
    return links


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("This file could not be opened as a PDF.") from exc
    try:
        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        if len(document) == 0:
            raise ValueError("The PDF has no pages.")
        pages = []
        for number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            if len(text) < 40 and OCR_ENABLED and page.get_images():
                try:
                    if not TESSDATA_DIR.is_dir():
                        raise RuntimeError(f"Tesseract language data was not found at {TESSDATA_DIR}")
                    textpage = page.get_textpage_ocr(
                        language="eng", dpi=150, full=True, tessdata=str(TESSDATA_DIR)
                    )
                    text = page.get_text("text", textpage=textpage).strip()
                except Exception as exc:
                    raise ValueError(
                        f"Page {number} needs OCR, but Tesseract is unavailable or OCR failed. "
                        "Install Tesseract or set OCR_ENABLED=false."
                    ) from exc
            external_links = extract_external_links(page)
            if external_links:
                text = f"{text}\n\nExternal links:\n" + "\n".join(external_links)
            pages.append((number, text))
        if not any(text for _, text in pages):
            raise ValueError("No readable text was found in this PDF.")
        return pages
    finally:
        document.close()


def chunk_pages(pages: list[tuple[int, str]], max_chars: int = 1200, overlap: int = 180):
    """Paragraph-aware chunks stay inside their source page."""
    for page, raw in pages:
        text = re.sub(r"[ \t]+", " ", raw).replace("\r", "")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            # Long paragraphs get split at whitespace rather than being discarded.
            words = paragraph.split()
            pieces = []
            piece = ""
            for word in words:
                if piece and len(piece) + len(word) + 1 > max_chars:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}".strip()
            if piece:
                pieces.append(piece)
            for part in pieces:
                if current and len(current) + len(part) + 2 > max_chars:
                    yield page, current
                    current = f"{current[-overlap:]} {part}".strip()
                else:
                    current = f"{current}\n\n{part}".strip()
                while len(current) > max_chars:
                    yield page, current[:max_chars]
                    current = current[max_chars - overlap:].strip()
        if current:
            yield page, current

"""Extract raw text from an uploaded CV file (PDF or DOCX)."""

import logging
from pathlib import Path
import pdfplumber
import docx

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """Route to the right extractor based on file extension.

    For PDFs, checks whether a text layer exists before falling back
    to OCR (not implemented here yet - raises so callers know to add
    an OCR step for scanned CVs, e.g. reusing the Qwen2-VL pipeline).
    """
    suffix = Path(file_path).suffix.lower()
    logger.info("Extracting text from %s (type=%s)", file_path, suffix)
    

    if suffix == ".pdf":
        text = _extract_pdf_text(file_path)
        logger.debug("Full extracted text from %s:\n%s", file_path, text)
    elif suffix == ".docx":
        text = _extract_docx_text(file_path)
        logger.debug("Full extracted text from %s:\n%s", file_path, text)
    else:
        logger.error("Unsupported file type: %s", suffix)
        raise ValueError(f"Unsupported file type: {suffix}")

    logger.info("Extracted %d characters from %s", len(text), file_path)
    return text


def _extract_pdf_text(file_path: str) -> str:
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)

    text = "\n".join(text_chunks).strip()

    if not text:
        logger.warning("No text layer found in %s - likely a scanned PDF", file_path)
        raise ValueError(
            "No extractable text layer found - this PDF is likely scanned "
            "and needs an OCR step before parsing."
        )
    return text


def _extract_docx_text(file_path: str) -> str:
    document = docx.Document(file_path)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs).strip()
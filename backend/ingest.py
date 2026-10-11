"""
Turns raw files (PDF / TXT / MD) into clean, overlapping text chunks —
each tagged with the tokens TF-IDF retrieval will score it against.
"""
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from pypdf import PdfReader

from . import config

TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Chunk:
    """One retrievable unit of text plus the metadata needed to cite it."""
    id: str
    source: str          # original filename
    chunk_index: int      # position of this chunk within the document
    text: str
    tokens: List[str] = field(default_factory=list)


def extract_text(file_path: Path) -> str:
    """Extract raw text from a PDF or plain-text/markdown file."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        return "\n".join(pages)

    if suffix in (".txt", ".md"):
        return file_path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def _clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping fixed-size chunks.

    A simple sliding window is used rather than a semantic splitter: it is
    easy to reason about, has no extra dependencies, and works well enough
    once combined with a reasonable chunk size and overlap. Where possible
    it ends a chunk on a paragraph or sentence boundary instead of cutting
    mid-sentence.
    """
    text = _clean_whitespace(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = max(end - overlap, start + 1)

    return chunks


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, and drop stopwords/short tokens."""
    words = TOKEN_RE.findall(text.lower())
    return [w for w in words if len(w) > 1 and w not in config.STOPWORDS]


def load_and_chunk(file_path: Path) -> List[Chunk]:
    """Full pipeline: read a file from disk and return a list of Chunks."""
    raw_text = extract_text(file_path)
    pieces = chunk_text(raw_text)

    return [
        Chunk(
            id=str(uuid.uuid4()),
            source=file_path.name,
            chunk_index=i,
            text=piece,
            tokens=tokenize(piece),
        )
        for i, piece in enumerate(pieces)
    ]

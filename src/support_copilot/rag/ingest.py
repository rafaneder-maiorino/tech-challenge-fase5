"""PDF -> cleaned page Documents -> chunks -> FAISS (in memory, cached on disk by settings hash)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from support_copilot.rag.config import RagSettings

logger = logging.getLogger(__name__)

_ICON_GLYPHS = re.compile(r"[-]")  # private-use icon font glyphs (e.g. the "Note" icon)
_SPACES = re.compile(r"[ \t]+")
_BREADCRUMB = re.compile(r"^\s*/\S[^\n]*?/\S[^\n]*$")  # "/Troubleshooting /Boot" printed atop each page


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def section_map(pdf_path: Path) -> dict[int, str]:
    """0-based page -> doc page name, from the outline written by scripts/build_manual.py."""
    reader = PdfReader(pdf_path)
    starts = sorted((reader.get_destination_page_number(item), item.title) for item in reader.outline)
    mapping: dict[int, str] = {}
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(reader.pages)
        for p in range(start, end):
            mapping[p] = title
    return mapping


def clean(text: str) -> str:
    text = _ICON_GLYPHS.sub("", text)
    text = text.replace("‣", ">")  # menu separator "System ‣ Settings" -> "System > Settings"
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if lines and _BREADCRUMB.match(lines[0]):  # navigation, already kept as `section` metadata
        lines = lines[1:]
    return "\n".join(lines)


def load_manual(settings: RagSettings) -> list[Document]:
    """LangChain PyPDFLoader, then cleaning and section metadata. Empty pages are dropped."""
    path = settings.manual_path
    actual = sha256_file(path)
    if actual != settings.manual_sha256:
        raise RuntimeError(f"manual hash {actual} != pinned {settings.manual_sha256} — rebuild or re-pin")
    sections = section_map(path)
    docs = []
    for doc in PyPDFLoader(str(path)).load():
        idx = int(doc.metadata["page"])
        text = clean(doc.page_content)
        if len(text) < 80:
            continue
        docs.append(Document(page_content=text, metadata={"page": idx + 1, "section": sections.get(idx, "unknown")}))
    logger.info("loaded %d non-empty pages from %s", len(docs), path)
    return docs


def section_title(section: str) -> str:
    """'troubleshooting/password_reset' -> 'Troubleshooting > Password reset'."""
    return " > ".join(part.replace("_", " ").capitalize() for part in section.split("/"))


def split(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Chunks carry a contextual header (the section title) in the EMBEDDED text, while the plain
    passage is kept in metadata['body'] for display. Removing the printed breadcrumb without this
    header cost hit@1 0.958 -> 0.875 (findings §4): the section name is signal, not noise."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
        c.metadata["body"] = c.page_content
        c.page_content = f"{section_title(str(c.metadata.get('section', '')))}\n{c.page_content}"
    return chunks


def embeddings(settings: RagSettings) -> Embeddings:
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return FastEmbedEmbeddings(model_name=settings.embedding_model, cache_dir=str(settings.model_cache_dir))


def build_index(chunks: list[Document], emb: Embeddings) -> FAISS:
    """Normalised vectors + inner product = cosine similarity as the FAISS score."""
    return FAISS.from_documents(chunks, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True)


def load_or_build(settings: RagSettings, emb: Embeddings | None = None) -> FAISS:
    emb = emb or embeddings(settings)
    path = settings.index_dir / settings.cache_key()
    if (path / "index.faiss").exists():
        # Safe: only ever loads an index this code wrote, addressed by a hash of its own inputs.
        store = FAISS.load_local(
            str(path),
            emb,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=True,
        )
        logger.info("index loaded from cache %s (%d vectors)", path, store.index.ntotal)
        return store
    t0 = time.perf_counter()
    chunks = split(load_manual(settings), settings.chunk_size, settings.chunk_overlap)
    store = build_index(chunks, emb)
    path.mkdir(parents=True, exist_ok=True)
    store.save_local(str(path))
    logger.info("index built: %d chunks in %.1fs -> %s", len(chunks), time.perf_counter() - t0, path)
    return store

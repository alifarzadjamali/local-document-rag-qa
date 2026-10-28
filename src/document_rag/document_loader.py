"""Load PDFs and turn their pages into retrieval-sized chunks."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def find_pdfs(directory: Path) -> list[Path]:
    """Return PDFs in a stable order so repeated runs are reproducible."""
    if not directory.exists():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: str(path).lower(),
    )


def load_pdfs(directory: Path) -> list[Document]:
    """Load each PDF page and attach simple, citation-friendly metadata."""
    pdf_paths = find_pdfs(directory)
    if not pdf_paths:
        raise ValueError(f"No PDF files found in {directory}")

    pages: list[Document] = []
    for path in pdf_paths:
        for page in PyPDFLoader(str(path)).load():
            if not page.page_content.strip():
                continue
            page.metadata["source"] = path.relative_to(directory).as_posix()
            page.metadata["path"] = str(path)
            pages.append(page)
    if not pages:
        raise ValueError(
            "The PDFs contain no selectable text. Scanned pages need OCR before indexing."
        )
    return pages


def split_documents(
    documents: list[Document], chunk_size: int = 1_000, chunk_overlap: int = 200
) -> list[Document]:
    """Split pages while preserving their source and page metadata."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_size must be positive and chunk_overlap must be between 0 and chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_split(
    directory: Path, chunk_size: int = 1_000, chunk_overlap: int = 200
) -> tuple[list[Document], int]:
    """Convenience entry point used by the ingestion command."""
    pages = load_pdfs(directory)
    return split_documents(pages, chunk_size, chunk_overlap), len(pages)

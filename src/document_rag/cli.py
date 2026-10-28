"""Command-line interface for indexing PDFs and asking questions."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from groq import APIError

from document_rag.agentic_rag import AgenticRAGPipeline
from document_rag.config import Settings
from document_rag.document_loader import find_pdfs, load_and_split
from document_rag.rag import RAGPipeline, create_llm
from document_rag.vector_store import index_documents, open_vector_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-rag", description="Learn RAG by indexing PDFs and asking grounded questions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Load, chunk, embed, and index PDF files.")
    ingest.add_argument("--documents", help="PDF directory (default: data/pdf).")

    files = subparsers.add_parser("files", help="List PDFs waiting in the document folder.")
    files.add_argument("--documents", help="PDF directory (default: data/pdf).")

    ask = subparsers.add_parser("ask", help="Ask a question about the indexed PDFs.")
    ask.add_argument("question", nargs="+", help="The question to ask.")
    ask.add_argument(
        "--agentic", action="store_true", help="Grade retrieval and retry once if needed."
    )
    ask.add_argument("--top-k", type=int, help="Number of chunks to retrieve.")

    search = subparsers.add_parser("search", help="Show retrieved passages without a Groq key.")
    search.add_argument("question", nargs="+", help="Search terms or a question.")
    search.add_argument("--top-k", type=int, help="Number of chunks to show.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "ingest":
        document_dir = settings.documents_dir if args.documents is None else Path(args.documents)
        try:
            chunks, page_count = load_and_split(
                document_dir, settings.chunk_size, settings.chunk_overlap
            )
        except (FileNotFoundError, ValueError) as error:
            raise SystemExit(str(error)) from None
        store = open_vector_store(
            settings.vector_store_dir, settings.collection_name, settings.embedding_model
        )
        chunk_count = index_documents(store, chunks)
        print(f"Indexed {chunk_count} chunks from {page_count} pages in {document_dir}.")
        return 0

    if args.command == "files":
        document_dir = settings.documents_dir if args.documents is None else Path(args.documents)
        try:
            pdfs = find_pdfs(document_dir)
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from None
        if not pdfs:
            print(f"No PDFs found in {document_dir}.")
        else:
            for pdf in pdfs:
                print(pdf.relative_to(document_dir).as_posix())
        return 0

    if args.command == "ask" and not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")

    top_k = settings.top_k if args.top_k is None else args.top_k
    if top_k < 1:
        raise SystemExit("--top-k must be at least 1")
    store = open_vector_store(
        settings.vector_store_dir, settings.collection_name, settings.embedding_model
    )
    if not store.get(limit=1)["ids"]:
        raise SystemExit("The index is empty. Run `document-rag ingest` first.")

    retriever = store.as_retriever(search_kwargs={"k": top_k})
    question = " ".join(args.question)
    if args.command == "search":
        documents = retriever.invoke(question)
        if not documents:
            print("No passages found. Try a different question or add more PDFs.")
        for number, document in enumerate(documents, start=1):
            page = document.metadata.get("page")
            source = document.metadata.get("source", "Unknown source")
            label = f"{source}, p. {int(page) + 1}" if page is not None else str(source)
            print(f"\n[{number}] {label}\n{document.page_content.strip()}")
        return 0

    llm = create_llm(settings.groq_model)
    pipeline = AgenticRAGPipeline(retriever, llm) if args.agentic else RAGPipeline(retriever, llm)
    try:
        result = pipeline.ask(question)
    except APIError as error:
        raise SystemExit(f"Groq request failed: {error}") from None

    print(f"\n{result.text}")
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source.label}")
    if args.agentic and pipeline.last_search_query != question:
        print(f"\nSearch query used: {pipeline.last_search_query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import pytest
from langchain_core.documents import Document

from document_rag import cli


def test_files_lists_nested_pdfs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "week1.PDF").touch()

    assert cli.main(["files", "--documents", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "notes/week1.PDF"


def test_search_shows_page_without_groq_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Store:
        def get(self, limit: int) -> dict[str, list[str]]:
            return {"ids": ["indexed"]}

        def as_retriever(self, search_kwargs: dict[str, int]) -> object:
            assert search_kwargs == {"k": 2}
            return self

        def invoke(self, question: str) -> list[Document]:
            assert question == "attention weights"
            return [
                Document(
                    page_content="Attention weights connect tokens.",
                    metadata={"source": "notes/week1.pdf", "page": 2},
                )
            ]

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(cli, "open_vector_store", lambda *args: Store())

    assert cli.main(["search", "attention", "weights", "--top-k", "2"]) == 0
    assert "[1] notes/week1.pdf, p. 3" in capsys.readouterr().out


def test_search_rejects_zero_top_k() -> None:
    with pytest.raises(SystemExit, match="--top-k must be at least 1"):
        cli.main(["search", "anything", "--top-k", "0"])

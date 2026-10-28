# Try it with your own PDFs

This is a one-hour practice session. Start with `data/pdf/Attention.pdf`, then repeat it with a
module handout or other PDF you can use from the University of Salford. Keep course materials on your
own computer if you do not have permission to share them.

## 1. Get the PDFs ready

The repo already has `Attention.pdf` and `Embedding.pdf` in `data/pdf/`. Check what the app sees:

```text
uv run document-rag files
```

To use a Salford PDF, copy it into `data/pdf/`. A subfolder such as `data/pdf/salford/` is fine too.
The `files` command should show it. PDFs with selectable text work best. If you can only select the
whole page as an image, run OCR on a copy first.

For a clean experiment with just one set of notes, put those PDFs in a separate folder and run
`uv run document-rag ingest --documents path/to/folder`. Each ingest rebuilds the index from that
folder. Running `files --documents path/to/folder` shows what will be included.

## 2. Build the index

```text
uv run document-rag ingest
```

Write down the page and chunk counts printed by the command. The first run may take a while because
the embedding model downloads. You do not need a Groq key for this step.

## 3. Check retrieval before asking the model

```text
uv run document-rag search "What is attention in a transformer?"
uv run document-rag search "How are input words represented?" --top-k 2
```

Read the displayed passages. Are they actually about the question? Note the filename and page number.
Try a question you expect the PDFs *not* to answer, such as "What is the library opening time?".
Nearest-neighbour search will still return something; the closest passage may be irrelevant.

For your own Salford notes, choose a fact you can verify by opening the PDF. Ask a question about a
definition, deadline, or worked example that is written there. Then try a question outside the PDF's
scope. Avoid uploading confidential or personal information.

## 4. Ask for an answer

Add `GROQ_API_KEY` to `.env` if you have one, then run:

```text
uv run document-rag ask "What is attention in a transformer?"
```

The answer should use numbered references such as `[1]`. The source list gives the PDF pages that
were retrieved. Open those pages and check the claim yourself. The source list shows retrieved pages;
it does not prove that every sentence in the answer is supported.

If the first search returned weak passages, try clearer wording. You can also compare the optional
retry path:

```text
uv run document-rag ask --agentic "How does it know which words matter?"
```

This path makes extra model calls, so use it as an experiment rather than the default.

## 5. Keep a tiny results log

Use three questions: one direct, one paraphrased, and one the PDF cannot answer. For each, record:

| Question | Useful passage in `search`? | Answer supported by the PDF? | What would you change? |
| --- | --- | --- | --- |
| Your direct question | | | |
| Your paraphrase | | | |
| Out-of-scope question | | | |

If retrieval is weak, try `--top-k 6` on both `search` and `ask`. If the passages are too broad,
change `CHUNK_SIZE` in `.env`, ingest again, and compare. Change one setting at a time so you can tell
what helped.

## 6. Follow one question through the code

Start in `src/document_rag/document_loader.py`, then read `vector_store.py`, `rag.py`, and `cli.py`.
Find where a PDF page becomes a chunk, where its filename and page number are kept, and where the
retrieved text is passed to the model. The tests in `tests/` show how these pieces can be checked
without calling Groq.

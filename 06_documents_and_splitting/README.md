# Lesson 06: Documents, Loaders & Text Splitting

This is the first of three lessons that build up to **RAG** (Retrieval-Augmented Generation).

```
 L06                         L07                        L08
 Load ──▶ Split ──▶ Embed ──▶ Store  ···  Retrieve ──▶ Prompt ──▶ LLM ──▶ Answer
 (files)  (chunks)  (vectors) (vector DB)  (top-k)
 └────────── indexing (done once, offline) ──────────┘ └──── querying (every question) ────┘
```

## Theory

### 1. The `Document` object
Everything in the RAG pipeline passes around `Document`s:
```python
Document(page_content="the text...", metadata={"source": "handbook.md", "page": 3})
```
- **`page_content`**: the text the LLM will eventually read.
- **`metadata`**: anything else, such as source file, page, author, date, or URL. It's used for **citations** and **filtering**
  ("only search HR documents from 2025").

### 2. Document loaders
A loader turns a source into `list[Document]`. Every loader implements `.load()`, which returns a list, and `.lazy_load()`, a generator for huge datasets.

> **⚠️ 2026 change: `langchain-community` is being sunset.** For years, most loaders (`TextLoader`, `CSVLoader`,
> `PyPDFLoader`, `WebBaseLoader`, …) lived in `langchain-community`. It now prints a `DeprecationWarning` and is no longer maintained.
> LangChain's official guidance is:
> 1. For simple sources, **write the loader yourself**. It's just "read the file, then build a `Document`" (see script 01).
> 2. For complex sources, use a **dedicated, maintained package**, such as a provider's own `langchain-<name>` package or the
>    underlying library directly (`pypdf`, `beautifulsoup4`, `unstructured`, …).
> 3. Subclass `langchain_core.document_loaders.BaseLoader` if you want a reusable loader that plugs into LangChain.

What you'll see in older code (all in `langchain_community.document_loaders`, now legacy):

| Source | Legacy loader | Modern approach |
|---|---|---|
| .txt / .md | `TextLoader` | `Path.read_text()` into a `Document` |
| .csv | `CSVLoader` (one Document per row) | The `csv` module, one `Document` per row |
| .pdf | `PyPDFLoader` (one Document per page) | `pypdf.PdfReader`, one `Document` per page |
| Web page | `WebBaseLoader` | `httpx` + `beautifulsoup4`, or an MCP tool |
| Folder | `DirectoryLoader` | `Path.glob()` or a custom `BaseLoader` |

### 3. Why split (chunk) documents?
1. **Context window**: a 300-page manual won't fit into one prompt, or it's expensive when it does.
2. **Retrieval precision**: an embedding of a whole book is a vague "average". An embedding of one paragraph
   is specific, so search finds the *exact* relevant passage.
3. **Less noise**: the LLM answers better from 3 focused paragraphs than from 30 pages.

### 4. How splitters work
**`RecursiveCharacterTextSplitter`** is the default choice. It tries to split on the "biggest" natural boundary first:
```
"\n\n" (paragraphs)  →  "\n" (lines)  →  " " (words)  →  "" (characters)
```
It only falls back to a smaller separator when a piece is still bigger than `chunk_size`. This keeps paragraphs and sentences whole wherever possible.

Key parameters:
- **`chunk_size`**: the maximum size of each chunk (in characters by default, or tokens with `from_tiktoken_encoder`).
- **`chunk_overlap`**: how much text is repeated between neighbouring chunks, so that a sentence cut at a boundary still appears whole in one of them.

```
chunk_size=100, chunk_overlap=20
|<------- chunk 1 ------->|
                     |<------- chunk 2 ------->|
                     ↑ 20 overlapping chars
```

### 5. Other splitters
| Splitter | Use for |
|---|---|
| `MarkdownHeaderTextSplitter` | Markdown. It splits on `#` headers and puts the header path in metadata. |
| `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` | Source code. It splits on `class`/`def` boundaries. |
| `HTMLHeaderTextSplitter` | HTML pages |
| Semantic chunkers (experimental) | Splitting where the *meaning* changes |

### 6. Choosing chunk size (a rule of thumb)
- FAQ or short facts: 200–500 characters
- General prose or docs: 500–1500 characters, with 10–20% overlap
- It depends on your data, so **test with real questions**, which you'll do in L08.

## Scripts
| File | Shows |
|---|---|
| `01_documents_and_loaders.py` | `Document`, DIY loaders for txt/csv, a custom `BaseLoader`, and the legacy community loader |
| `02_text_splitters.py` | Recursive splitting, overlap visualised, and the Markdown and code splitters |

## Pitfalls
- Chunks too small: they lose context ("It costs Rs 49", but *what* costs Rs 49?).
- Chunks too large: retrieval gets vague and prompts get expensive.
- Losing metadata: always keep `source` so you can cite answers.
- PDFs with tables or scans load badly. Consider `unstructured` or OCR.

## Exercises
1. Split `employee_handbook.md` with `chunk_size=100` and `chunk_size=1000`. Read a few chunks from each. Which would you rather retrieve?
2. Use `MarkdownHeaderTextSplitter` and then a `RecursiveCharacterTextSplitter` on its output, which is a common two-stage pattern.
3. `uv pip install pypdf`, put any PDF in `data/`, and write a `PdfLoader(BaseLoader)` that yields one Document per page:
   ```python
   from pypdf import PdfReader
   for i, page in enumerate(PdfReader(path).pages):
       yield Document(page_content=page.extract_text(), metadata={"source": path, "page": i})
   ```

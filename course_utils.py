"""
Shared helpers for every lesson in this course.

Every script does:
    from course_utils import get_model, get_embeddings, banner

so you can switch the WHOLE course to a different LLM provider by editing
one line in `.env` instead of touching every file.

Model strings use LangChain's "provider:model" format, e.g.
    ollama:llama3.1                 (free, local - the default)
    anthropic:claude-sonnet-5       (needs ANTHROPIC_API_KEY)
    openai:gpt-4.1-mini             (needs OPENAI_API_KEY + `pip install langchain-openai`)
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

# Load variables from the .env file in the course root (if it exists).
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

COURSE_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(COURSE_ROOT, "data")  # sample documents used from lesson 06

CHAT_MODEL = os.getenv("LC_CHAT_MODEL", "ollama:llama3.1")
EMBED_MODEL = os.getenv("LC_EMBED_MODEL", "ollama:nomic-embed-text")


def get_model(**kwargs):
    """Return a chat model. kwargs (temperature, max_tokens, ...) pass straight through."""
    kwargs.setdefault("temperature", 0)  # deterministic answers are easier to learn from
    return init_chat_model(CHAT_MODEL, **kwargs)


def get_embeddings():
    """Return an embeddings model (turns text into vectors). Used from lesson 07 onward."""
    return init_embeddings(EMBED_MODEL)


def banner(title: str):
    """Print a visible section header so script output is easy to follow."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_bakery_chunks():
    """
    Load and chunk the sample bakery data (built step by step in lesson 06).
    Returns a list of Documents ready to embed. Used by lessons 07, 08 and 13.
    """
    import csv
    from pathlib import Path

    from langchain_core.documents import Document
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    data = Path(DATA_DIR)
    chunks = []

    # 1. Handbook: split on markdown headers so each chunk is one policy section
    md = (data / "employee_handbook.md").read_text(encoding="utf-8")
    for d in MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")]).split_text(md):
        section = d.metadata.get("section", "Introduction")
        chunks.append(Document(page_content=f"{section}\n{d.page_content}",
                               metadata={"source": "employee_handbook.md", "section": section, "audience": "staff"}))

    # 2. FAQ: split on blank lines so each chunk is exactly one Q&A pair.
    #    (A plain str.split is enough here. A text splitter would MERGE small Q&As up to chunk_size.)
    faq = (data / "customer_faq.txt").read_text(encoding="utf-8")
    for text in faq.split("\n\n")[1:]:  # [1:] skips the title line
        chunks.append(Document(page_content=text.strip(),
                               metadata={"source": "customer_faq.txt", "audience": "customer"}))

    # 3. Products: one row per chunk
    with open(data / "products.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chunks.append(Document(page_content="\n".join(f"{k}: {v}" for k, v in row.items()),
                                   metadata={"source": "products.csv", "audience": "customer",
                                             "category": row["category"]}))
    return chunks


def get_bakery_store():
    """
    Return an InMemoryVectorStore of the bakery chunks (lesson 07).
    Re-uses data/bakery_index_<model>.json if it exists so we don't re-embed on every run.
    Delete that file if you change the data or the embedding model.
    """
    from langchain_core.vectorstores import InMemoryVectorStore

    # The file name includes the embedding model: vectors from different models are not comparable.
    safe_name = EMBED_MODEL.replace(":", "_").replace("/", "_")
    path = os.path.join(DATA_DIR, f"bakery_index_{safe_name}.json")
    if os.path.exists(path):
        return InMemoryVectorStore.load(path, embedding=get_embeddings())
    store = InMemoryVectorStore.from_documents(load_bakery_chunks(), embedding=get_embeddings())
    store.dump(path)
    return store

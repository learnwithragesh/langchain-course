"""
Lesson 06.1: The Document object and document loaders.

NOTE (2026): `langchain-community`, which held most classic loaders like TextLoader and CSVLoader,
is being SUNSET. LangChain's guidance is to write simple loaders in your own code
(they're just "read file -> Document") or use a dedicated integration package.
We do both here, and show the legacy community loader at the end because old code uses it everywhere.
"""

import csv
import warnings
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from course_utils import DATA_DIR, banner

data = Path(DATA_DIR)

banner("A Document is just text + metadata")
doc = Document(page_content="LangChain was released in October 2022.", metadata={"source": "notes", "topic": "history"})
print(repr(doc))

banner("A text/markdown 'loader' is 1 line of plain Python")
faq_path = data / "customer_faq.txt"
faq = Document(page_content=faq_path.read_text(encoding="utf-8"), metadata={"source": faq_path.name})
print(f"{len(faq.page_content)} characters, metadata={faq.metadata}")
print("first 150 chars:\n", faq.page_content[:150])

banner("CSV: one ROW -> one Document (with columns as metadata for filtering)")
products = []
with open(data / "products.csv", encoding="utf-8") as f:
    for i, row in enumerate(csv.DictReader(f)):
        text = "\n".join(f"{k}: {v}" for k, v in row.items())
        products.append(Document(page_content=text, metadata={
            "source": "products.csv", "row": i, "category": row["category"], "eggless": row["eggless"] == "yes"}))
print(f"{len(products)} documents")
print("row 0 page_content:\n" + products[0].page_content)
print("row 0 metadata:", products[0].metadata)


banner("A reusable loader class: subclass BaseLoader, implement lazy_load()")


class FolderLoader(BaseLoader):
    """Loads every .md/.txt file in a folder as one Document each."""

    def __init__(self, folder: str, patterns=("*.md", "*.txt")):
        self.folder = Path(folder)
        self.patterns = patterns

    def lazy_load(self) -> Iterator[Document]:  # a generator: one file at a time, memory-friendly
        for pattern in self.patterns:
            for path in sorted(self.folder.glob(pattern)):
                yield Document(page_content=path.read_text(encoding="utf-8"),
                               metadata={"source": path.name, "type": path.suffix})


loader = FolderLoader(DATA_DIR)
all_docs = loader.load()  # .load() comes free from BaseLoader: list(lazy_load())
for d in all_docs:
    print(f"  {d.metadata['source']:25} {len(d.page_content):6} chars")

banner("Adding your own metadata (useful for filtering later)")
for d in all_docs:
    d.metadata["department"] = "HR" if "handbook" in d.metadata["source"] else "Customer Service"
print([(d.metadata["source"], d.metadata["department"]) for d in all_docs])

banner("LEGACY: langchain_community loaders (still work, but sunset)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langchain_community.document_loaders import TextLoader
legacy = TextLoader(str(faq_path), encoding="utf-8").load()
print(f"TextLoader -> {len(legacy)} doc, metadata={legacy[0].metadata}")
print("Same result as our 1-line version. For PDFs, web pages and so on, look for a maintained package")
print("(e.g. `pypdf` directly, or a provider's own langchain-<name> package).")

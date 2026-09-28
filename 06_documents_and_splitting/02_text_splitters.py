"""
Lesson 06.2: Splitting documents into chunks.
"""

import os

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from course_utils import DATA_DIR, banner

text = ("LangChain helps you build LLM apps. It provides prompts, models and parsers.\n\n"
        "RAG lets a model answer questions from your own documents. First you split documents into chunks. "
        "Then you embed each chunk and store it in a vector database.\n\n"
        "Agents let the model call tools in a loop until the task is done.")

banner("RecursiveCharacterTextSplitter: splits at paragraphs, then lines, then words")
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
for i, chunk in enumerate(splitter.split_text(text)):
    print(f"  [{i}] ({len(chunk):3} chars) {chunk!r}")

banner("Overlap visualised: the end of one chunk repeats at the start of the next")
long_sentence = " ".join(f"word{i}" for i in range(40))
for chunk in RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=25).split_text(long_sentence):
    print("  ", chunk)

banner("CharacterTextSplitter: only ONE separator, less smart")
naive = CharacterTextSplitter(separator=" ", chunk_size=100, chunk_overlap=0)
for i, chunk in enumerate(naive.split_text(text)):
    print(f"  [{i}] {chunk!r}")

banner("Splitting Documents keeps (and copies) metadata")
faq_path = os.path.join(DATA_DIR, "customer_faq.txt")
docs = [Document(page_content=open(faq_path, encoding="utf-8").read(), metadata={"source": "customer_faq.txt"})]
chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50,
                                        add_start_index=True).split_documents(docs)
print(f"1 document -> {len(chunks)} chunks")
for c in chunks[:3]:
    print(f"\n  metadata={c.metadata}\n  {c.page_content[:120]!r}...")

banner("MarkdownHeaderTextSplitter: the header path goes into metadata")
md = open(os.path.join(DATA_DIR, "employee_handbook.md"), encoding="utf-8").read()
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")])
md_chunks = md_splitter.split_text(md)
for c in md_chunks:
    print(f"  {c.metadata.get('section', '(intro)'):28} {len(c.page_content):4} chars")

banner("Code-aware splitting (Python)")
code = '''
class Oven:
    def preheat(self, temp):
        self.temp = temp

    def bake(self, item, minutes):
        return f"{item} baked for {minutes} min at {self.temp}C"


def make_bread():
    oven = Oven()
    oven.preheat(200)
    return oven.bake("bread", 30)
'''
py_splitter = RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=120, chunk_overlap=0)
for i, c in enumerate(py_splitter.split_text(code)):
    print(f"--- chunk {i} ---\n{c}")

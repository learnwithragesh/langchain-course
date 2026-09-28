"""
Lesson 07.3: Retrievers. The Runnable interface for 'get me relevant documents'.
"""

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore

from course_utils import banner, get_embeddings, load_bakery_chunks

store = InMemoryVectorStore.from_documents(load_bakery_chunks(), embedding=get_embeddings())

banner("as_retriever(): a store becomes a Runnable (str -> list[Document])")
retriever = store.as_retriever(search_kwargs={"k": 2})
print("type:", type(retriever).__name__)
for d in retriever.invoke("What time do stores close on Tuesday?"):
    print("  ", d.page_content[:100].replace("\n", " | "))

banner("Retrievers are Runnables, so batch works")
results = retriever.batch(["loyalty points", "wedding cake notice"])
for q, docs in zip(["loyalty points", "wedding cake notice"], results):
    print(f"  {q!r:24} -> {docs[0].page_content[:60]!r}")

banner("Other search types")
# MMR = relevant but diverse results
mmr = store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.25})
print("mmr:", [d.page_content.splitlines()[0][:40] for d in mmr.invoke("bread")])


# search_type="similarity_score_threshold" exists, but InMemoryVectorStore raises NotImplementedError for it.
# (Chroma, FAISS, pgvector and others support it.) Knowing the building blocks, we can write it ourselves:


@chain
def threshold_retriever(query: str) -> list[Document]:
    """Return up to 5 chunks, but only those scoring above 0.6 cosine similarity."""
    return [doc for doc, score in store.similarity_search_with_score(query, k=5) if score >= 0.6]


print("threshold 0.6, off-topic query:", threshold_retriever.invoke("Who won the football world cup?"))
print("threshold 0.6, on-topic query :", len(threshold_retriever.invoke("Is anything gluten free?")), "docs")
print("-> an empty result lets your app say 'I don't know' instead of guessing.")

banner("Another custom retriever: keyword search")


@chain
def keyword_retriever(query: str) -> list[Document]:
    """Dumb keyword search: returns chunks containing ANY query word (len>3)."""
    words = [w.lower() for w in query.split() if len(w) > 3]
    return [d for d in load_bakery_chunks() if any(w in d.page_content.lower() for w in words)][:3]


for d in keyword_retriever.invoke("refund policy"):
    print("  ", d.page_content[:80].replace("\n", " | "))
print("Notice 'Leave Policy' matched only because it contains the word 'policy'. Keyword search has no idea of meaning.")
print("\nIdea: combining keyword + vector results = 'hybrid search' (production RAG does this).")

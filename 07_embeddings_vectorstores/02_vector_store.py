"""
Lesson 07.2: A vector store over our bakery documents.
"""

import os
import time

from langchain_core.vectorstores import InMemoryVectorStore

from course_utils import COURSE_ROOT, banner, get_embeddings, load_bakery_chunks

emb = get_embeddings()
chunks = load_bakery_chunks()

banner(f"Indexing {len(chunks)} chunks")
start = time.time()
store = InMemoryVectorStore(embedding=emb)
ids = store.add_documents(chunks)  # embeds every chunk and stores (vector, doc)
print(f"indexed {len(ids)} chunks in {time.time() - start:.1f}s")

banner("similarity_search: top-k nearest chunks")
for doc in store.similarity_search("How many holidays do employees get?", k=2):
    print(f"  [{doc.metadata['source']}] {doc.page_content[:90]!r}")

banner("similarity_search_with_score: see HOW similar")
for doc, score in store.similarity_search_with_score("Is there anything without gluten?", k=4):
    print(f"  {score:.3f}  {doc.page_content[:80]!r}")

banner("Metadata filter: only search staff documents")
# InMemoryVectorStore takes a Python function. Real DBs take a dict like {"audience": "staff"}.
for doc in store.similarity_search("discount", k=3, filter=lambda d: d.metadata["audience"] == "staff"):
    print(f"  [{doc.metadata['audience']}] {doc.page_content[:80]!r}")

banner("Similarity vs MMR for a broad query: 'cake'")
print("similarity:")
for doc in store.similarity_search("cake", k=4):
    print("   ", doc.page_content.splitlines()[0][:70])
print("MMR, lambda_mult=0.25 (relevant AND diverse):")
# lambda_mult: 1.0 = pure relevance (same as similarity), 0.0 = maximum diversity. The default is 0.5.
for doc in store.max_marginal_relevance_search("cake", k=4, fetch_k=15, lambda_mult=0.25):
    print("   ", doc.page_content.splitlines()[0][:70])

banner("Persist the index to disk so we don't re-embed every run")
path = os.path.join(COURSE_ROOT, "data", "bakery_index_demo.json")
store.dump(path)
print(f"saved -> {path} ({os.path.getsize(path) // 1024} KB)")
reloaded = InMemoryVectorStore.load(path, embedding=emb)
print("reloaded; search still works:", reloaded.similarity_search("delivery fee", k=1)[0].page_content[:60])

"""
Lesson 07.1: Embeddings. Turning meaning into numbers.
"""

import math

from course_utils import EMBED_MODEL, banner, get_embeddings

emb = get_embeddings()

banner(f"Embedding a sentence with {EMBED_MODEL}")
v = emb.embed_query("I love freshly baked bread.")
print(f"vector length: {len(v)}")
print(f"first 8 values: {[round(x, 3) for x in v[:8]]}")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


banner("Cosine similarity matrix: similar meanings score higher")
sentences = [
    "How do I get my money back?",
    "What is your refund policy?",
    "Do you deliver to my house?",
    "Can you bring the cake to my home?",
    "The stock market fell today.",
]
vectors = emb.embed_documents(sentences)  # batch-embed many texts

print(" " * 38 + "".join(f"  S{i}  " for i in range(len(sentences))))
for i, (s, vi) in enumerate(zip(sentences, vectors)):
    row = "".join(f" {cosine(vi, vj):5.2f}" for vj in vectors)
    print(f"S{i} {s[:34]:34} {row}")

banner("Semantic search beats keyword search")
query = "money back"
print(f"Query: {query!r}")
q = emb.embed_query(query)
ranked = sorted(zip(sentences, vectors), key=lambda p: cosine(q, p[1]), reverse=True)
for s, vec in ranked:
    keyword_hit = any(w in s.lower() for w in query.split())
    print(f"  {cosine(q, vec):.3f}  keyword match={str(keyword_hit):5}  {s}")
print("\nNotice 'refund policy' ranks high although it shares NO words with 'money back'.")

# Lesson 07: Embeddings & Vector Stores

## Theory

### 1. What is an embedding?
An **embedding model** turns text into a **vector**: a list of numbers (768 of them for `nomic-embed-text`)
that represents the text's **meaning**.

```
"How do I get a refund?"     → [0.12, -0.45, 0.88, ..., 0.03]   ┐ close together
"What is your return policy?" → [0.10, -0.41, 0.91, ..., 0.05]   ┘ (similar meaning)
"Masala croissant price"      → [-0.67, 0.22, -0.10, ..., 0.54]    far away
```

Texts with similar meanings produce vectors that point in similar directions, **even when they share no words**.
This is **semantic search**, and it's why RAG works much better than keyword search (Ctrl+F).

### 2. Measuring similarity
**Cosine similarity** is the cosine of the angle between two vectors:
- `1.0` means the same direction (the same meaning)
- `~0` means unrelated
- negative means opposite (rare with text)

```
cos(A, B) = (A · B) / (|A| × |B|)
```

### 3. Embedding models vs. chat models
| | Chat model | Embedding model |
|---|---|---|
| Input | messages | text |
| Output | text | a fixed-size vector |
| Used for | generating answers | search, clustering, deduplication |
| Examples | llama3.1, Claude | nomic-embed-text, OpenAI text-embedding-3, Voyage, Cohere embed |

In LangChain, `Embeddings` has two methods:
- `embed_query(text)`: embeds one search query
- `embed_documents([texts])`: embeds many documents (in batches)

(Some models embed queries and documents slightly differently, which is why there are two methods.)

> Anthropic doesn't provide an embedding model. Claude users typically pair it with Voyage AI, OpenAI, or a local model.
> **Rule:** always use the *same* embedding model to index and to query. Vectors from different models aren't comparable.

### 4. Vector stores
A **vector store** (vector database) saves `(vector, Document)` pairs and quickly finds the nearest vectors to a query vector.

| Vector store | Type | Package |
|---|---|---|
| `InMemoryVectorStore` | In RAM, for learning and prototypes | `langchain-core` (built in) |
| Chroma | Local file or server | `langchain-chroma` |
| FAISS | Local, very fast, from Meta | `faiss-cpu` |
| pgvector | Postgres extension | `langchain-postgres` |
| Pinecone, Qdrant, Weaviate, Milvus | Managed or scalable | `langchain-pinecone`, `langchain-qdrant`, … |

They all share the same interface, so switching is usually a one-line change:
```python
store.add_documents(docs)
store.similarity_search("query", k=4)
store.similarity_search_with_score("query", k=4)
store.max_marginal_relevance_search("query", k=4, fetch_k=20)
store.as_retriever(search_kwargs={"k": 4})
```

### 5. Search types
- **Similarity (top-k)**: returns the k nearest chunks.
- **MMR (Maximal Marginal Relevance)**: returns chunks that are relevant *and* different from each other, which avoids 4 near-duplicate results.
  `lambda_mult` controls the balance: 1.0 is pure relevance and 0.0 is maximum diversity. With the default of 0.5, our small dataset gave
  the *same* results as plain similarity, and 0.25 made the difference visible.
- **Metadata filter**: restricts the search to documents that match a condition (for example, `audience == "staff"`).
- **Score threshold**: only returns results above a minimum similarity. This isn't supported by `InMemoryVectorStore`, so script 03 builds it by hand.

### 6. Retrievers
A **retriever** is a Runnable that takes a `str` query and returns `list[Document]`.
`vector_store.as_retriever()` creates one. Because it's a Runnable, it plugs straight into LCEL chains (L08).
Retrievers don't *have* to use vectors. A retriever can wrap keyword search (BM25), a SQL query, or a web search API.

## Scripts
| File | Shows |
|---|---|
| `01_embeddings.py` | Embedding text, cosine similarity by hand, and semantic vs. keyword matching |
| `02_vector_store.py` | Indexing the bakery data, similarity search, scores, filters, MMR, and saving to disk |
| `03_retrievers.py` | `as_retriever()`, search types, and retrievers as Runnables |

## Pitfalls
- Mixing embedding models between indexing and querying.
- Re-embedding everything on every run. Persist the index (script 02 shows `dump`/`load`).
- Treating scores as absolute truth. Scores vary per model, so compare them relatively.
- Semantic search is weak on exact identifiers ("SKU-8841"). Combine it with keyword search (**hybrid search**).

## Exercises
1. In `01_embeddings.py`, add a sentence in Tamil or Hindi with the same meaning as an English one. Is the similarity high? (That depends on whether the model is multilingual.)
2. Search for "can my mom use my staff discount?" Which chunk comes first? Is it the right one?
3. Compare `k=2` with `k=6`, and similarity with MMR, for the query "cake".

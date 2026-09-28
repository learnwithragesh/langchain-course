# Lesson 08: RAG (Retrieval-Augmented Generation)

## Theory

### 1. The problem RAG solves
An LLM knows only what was in its training data. It doesn't know:
- **private** information (your company handbook, your code, your emails)
- **recent** information (anything after its training cutoff)

Ask it anyway and it will often **hallucinate** a confident, plausible, wrong answer.

**Fine-tuning** (re-training the model on your data) is expensive and slow, and it still hallucinates.
**RAG** is simpler: *look up the relevant text first, then give it to the model with the question.*

> It's like the difference between a *closed-book* exam (the plain LLM) and an *open-book* exam (RAG).

### 2. The pipeline

```
                       ┌─────────────────── INDEXING (once) ───────────────────┐
                       │  docs ─▶ split ─▶ embed ─▶ vector store                │
                       └────────────────────────────────────────────────────────┘

QUERY TIME (every question):
 "How many leave days?"
        │
        ├──────────────▶ retriever ──▶ top-k chunks ──┐
        │                                             ▼
        └──────────────────────────────────────▶  PROMPT:
                                                   "Answer ONLY from this context:
                                                    {context}
                                                    Question: {question}"
                                                         │
                                                         ▼
                                                        LLM ──▶ "18 days of paid leave…"
```

In LCEL, this is almost exactly the diagram from lesson 04:
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

### 3. The RAG prompt matters
A good RAG prompt:
1. says to use **only** the provided context
2. says what to do when the answer is **not** in the context ("say you don't know")
3. asks for **citations** (the source file or section)
4. keeps the context clearly separated (with XML tags or delimiters)

### 4. Benefits
- It answers questions about private or new data without retraining.
- It **reduces hallucination** (though it doesn't eliminate it).
- It supports **citations**, so users can verify answers.
- Updating knowledge is easy: re-index the changed documents.

### 5. What goes wrong, and how to fix it
| Symptom | Likely cause | Fix |
|---|---|---|
| "I don't know" when the answer exists | Retrieval missed the chunk | Better chunking, higher k, hybrid search, query rewriting |
| Wrong answer with the right chunks | The model ignored or misread the context | A stronger model, a clearer prompt, fewer and more relevant chunks |
| Follow-up questions fail ("what about sick leave?") | The retriever can't see the chat history | **Condense** the question into a standalone query first (script 03) |
| The answer mixes sources up | Chunks lack context | Add section titles or metadata into the chunk text |

**Always debug retrieval first.** Print the retrieved chunks. If the right text isn't there, no prompt can fix it.

### 6. Advanced RAG techniques (for later)
- **Query rewriting / multi-query**: the LLM rewrites the question, or generates several versions of it, before retrieval.
- **Hybrid search**: combine keyword (BM25) and vector results.
- **Re-ranking**: retrieve 20 chunks, then use a cross-encoder or reranker model to keep the best 4.
- **Parent-document retrieval**: search small chunks, but return the larger parent section.
- **Agentic RAG**: give the retriever to an **agent as a tool**, so it decides *when* and *what* to search (L10, L13).

### 7. Evaluating RAG
Measure two things separately:
1. **Retrieval quality**: did the right chunk appear in the top-k? (hit rate)
2. **Answer quality**: is the answer correct and faithful to the context? (keyword checks, LLM-as-judge, or LangSmith evaluations)

Script 04 builds a tiny evaluation harness.

## Scripts
| File | Shows |
|---|---|
| `01_basic_rag.py` | No-RAG vs. RAG side by side, and the LCEL RAG chain |
| `02_rag_with_sources.py` | Returning the answer **and** the sources used |
| `03_conversational_rag.py` | Follow-up questions: condense the question, retrieve, then answer |
| `04_evaluate_rag.py` | A mini test set: retrieval hit rate and answer correctness |

## Exercises
1. Ask `01_basic_rag.py` something that's *not* in the data ("Do you sell pizza?"). Does it admit it doesn't know?
2. Change `k` from 3 to 1 in `04_evaluate_rag.py`. How do the scores change?
3. Add 3 more questions to the evaluation set, including one tricky one.
4. Add a new section to `data/employee_handbook.md`, delete the cached `data/bakery_index_*.json`, and ask about the new section.

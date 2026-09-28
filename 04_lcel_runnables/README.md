# Lesson 04: LCEL (LangChain Expression Language) & Runnables

## Theory

### 1. The Runnable protocol
Everything you have used so far (prompts, models, and parsers) is a **Runnable**.
A Runnable is anything with `invoke / stream / batch` (plus the async versions). Because they share one interface,
you can **snap them together like LEGO**.

**LCEL** is the syntax for snapping them together. It's just the `|` operator:

```python
chain = prompt | model | parser
```
This builds a `RunnableSequence`. Calling `chain.invoke(x)` does:
`parser.invoke(model.invoke(prompt.invoke(x)))`.

You get these for free on *every* chain:
- streaming (tokens flow through the whole pipeline)
- batching and async
- retries and fallbacks
- tracing in LangSmith
- a printable graph of the chain

### 2. The building blocks

| Runnable | What it does | Shortcut |
|---|---|---|
| `RunnableSequence` | Run steps one after another | `a \| b \| c` |
| `RunnableParallel` | Run steps **at the same time** on the same input and return a dict | A plain `dict` inside a chain: `{"x": a, "y": b}` |
| `RunnablePassthrough` | Pass the input through unchanged | |
| `RunnablePassthrough.assign(k=fn)` | Keep the input dict **and add** new key(s) | |
| `RunnableLambda(fn)` | Wrap any Python function as a step | A plain function inside a chain |
| `itemgetter("key")` | Pick one key out of a dict | (from Python's `operator` module) |
| routing | Choose the next step at runtime | A function that *returns* a Runnable |

### 3. Data-flow diagram of a typical LCEL chain

```
input: {"question": "..."}
   │
   ▼
RunnableParallel ────────────────┐
   │ "context":  retriever       │   (both run at the same time)
   │ "question": passthrough     │
   └─────────────────────────────┘
   │  {"context": [...], "question": "..."}
   ▼
prompt ──▶ model ──▶ StrOutputParser ──▶ "answer"
```
This exact shape is RAG, which you'll build in lesson 08.

### 4. Coercion: the magic you should understand
Inside a chain:
- a **dict** becomes a `RunnableParallel`
- a **function** becomes a `RunnableLambda`

So `{"a": fn1, "b": fn2} | prompt` is valid LCEL. At least one side of `|` must already be a Runnable.

### 5. Reliability helpers
- `.with_retry(stop_after_attempt=3)`: retries on exceptions, such as flaky network calls.
- `.with_fallbacks([backup])`: if the primary fails, it tries the backup (for example, a cloud model falling back to a local one).
- `.with_config(run_name=..., tags=[...])`: labels runs in traces.
- `invoke(x, config={"max_concurrency": 5})`: limits parallelism in `batch`.

### 6. When NOT to use LCEL
LCEL is good for **linear or branching pipelines**. For **loops, cycles, long-running state, or agents**
use `create_agent` (L10) or LangGraph (L12). As a rule of thumb: if you need a `while` loop, don't use LCEL.

## Scripts
| File | Shows |
|---|---|
| `01_pipe_basics.py` | Sequences, functions in chains, and chaining two LLM calls |
| `02_parallel_and_passthrough.py` | `RunnableParallel`, `RunnablePassthrough.assign`, and `itemgetter` |
| `03_routing.py` | Classify, then route to a specialised chain |
| `04_reliability_and_inspection.py` | Retries, fallbacks, config, and drawing the chain graph |

## Pitfalls
- `a | b` where *both* are plain functions or dicts is a Python error. Wrap one side in `RunnableLambda`.
- Passing a `str` where the next step expects a `dict` (and the reverse). Print the intermediate outputs.
- A `RunnableLambda` function must take **exactly one argument** (the input). Use a dict for multiple values.
- Asking an LLM to do character-level work. In `04_reliability...py`, llama3.1 reversed "stressed" as `dessetS`.
  Models see **tokens**, not letters. Do string manipulation in plain Python (a `RunnableLambda`), not with the LLM.

## Exercises
1. Build a chain that (1) generates a short story from a `{topic}` and (2) generates a title for that story. Return both.
2. Add a third route, "history", to `03_routing.py`.
3. Create a fallback from a non-existent model `ollama:does-not-exist` to `ollama:llama3.1` and confirm it works.

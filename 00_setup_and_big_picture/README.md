# Lesson 00: Setup & the Big Picture

## 1. What is an LLM, really?

A **Large Language Model** (LLM) is a neural network trained to do one thing: *given some text, predict the next token*.
A **token** is a word piece, roughly 3/4 of an English word. By predicting one token at a time, again and again, it produces sentences, code, and JSON.

Key facts that shape everything in LangChain:

| Fact about LLMs | Consequence | LangChain feature that helps |
|---|---|---|
| Input is text in, text out | You must turn your task into a prompt | Prompt templates (L02) |
| Output is free-form text | Your code needs structured data | Structured output (L03) |
| **Stateless**: no memory between calls | Chatbots forget everything | Memory (L05) |
| Knowledge is frozen at training time and has no private data | It can't answer about *your* docs | RAG (L06–08) |
| It can only *write text*, not *do* things | It can't check the weather or query a DB | Tools & Agents (L09–11) |
| There is a limited **context window** (max tokens per call) | You can't paste in a whole book | Splitting, retrieval, summarization |
| Output is non-deterministic | Results vary | `temperature=0`, evaluation, tracing |

## 2. What is LangChain?

**LangChain is a framework for building applications powered by LLMs.** It gives you:

1. **A standard interface to every model provider.** The same code talks to Ollama, Claude, GPT, and Gemini.
   You change `"ollama:llama3.1"` to `"anthropic:claude-sonnet-5"` and nothing else.
2. **Building blocks**: prompts, output parsers, document loaders, text splitters, embeddings, vector stores, and tools.
3. **A way to compose them**: LCEL, the `|` pipe (L04).
4. **Agents**: `create_agent()` lets the LLM decide which tools to call in a loop (L10).

Think of it this way: **the LLM is the engine, and LangChain is the car around it** (steering, fuel line, dashboard).

## 3. The ecosystem (the packages you'll see)

```
┌────────────────────────────────────────────────────────────────────┐
│  LangSmith   (web platform: tracing, debugging, evaluation)        │
├────────────────────────────────────────────────────────────────────┤
│  langchain        high-level: create_agent, middleware,            │
│                   init_chat_model, init_embeddings                 │
├────────────────────────────────────────────────────────────────────┤
│  langgraph        low-level runtime: stateful graphs, persistence, │
│                   human-in-the-loop. create_agent is BUILT on it.  │
├────────────────────────────────────────────────────────────────────┤
│  langchain-core   the base abstractions: Runnable, messages,       │
│                   prompts, parsers, tools, vector store interface  │
├────────────────────────────────────────────────────────────────────┤
│  Integrations     langchain-ollama, langchain-anthropic,           │
│                   langchain-openai, langchain-community (loaders)  │
└────────────────────────────────────────────────────────────────────┘
```

- **langchain-core**: the tiny, stable foundation. Everything implements the `Runnable` interface.
- **langchain-<provider>**: one package per provider (we use `langchain-ollama` and `langchain-anthropic`).
- **langchain**: the convenient top-level package. In v1.0 (Oct 2025) it was slimmed down to focus on agents.
  Old things like `LLMChain` moved to `langchain-classic`.
- **langgraph**: for when you need full control of the flow (loops, branches, persistence). Covered in L12.
- **LangSmith**: an optional hosted service that records every step your app takes. It's very useful for debugging.

## 4. The mental model for the whole course

```
         ┌──────────┐    ┌───────┐    ┌────────┐
input ──▶│  Prompt  │──▶│ Model │──▶│ Parser │──▶ output
         └──────────┘    └───────┘    └────────┘
              ▲               │
              │          (tool calls)
       ┌──────┴─────┐         ▼
       │ Retriever  │     ┌───────┐
       │ (your docs)│     │ Tools │
       └────────────┘     └───────┘
```

Every LangChain app is some combination of these boxes. Each lesson adds one box.

## 5. Setup (already done for you)

This folder was already set up with the following steps. `setup.sh` repeats them if you ever need to rebuild:

```bash
uv venv --python 3.12 .venv                       # isolated Python environment
uv pip install --python .venv/bin/python -r requirements.txt
echo "$PWD" > .venv/lib/python3.12/site-packages/langchain_course.pth   # lets lessons import course_utils
ollama pull llama3.1 && ollama pull nomic-embed-text
```

### Choosing a model

`course_utils.py` reads `LC_CHAT_MODEL` from `.env`. The default is `ollama:llama3.1`.

| Option | Cost | Quality | Setup |
|---|---|---|---|
| `ollama:llama3.1` (8B, local) | Free | OK. It sometimes needs very explicit prompts | Already installed |
| `ollama:qwen3.6:27b` (local) | Free | Much better, but slower and uses about 17 GB of RAM | Already installed |
| `anthropic:claude-sonnet-5` | Paid per token | Excellent | `cp .env.example .env`, then add `ANTHROPIC_API_KEY` |

> **Learning tip:** a small local model is actually a *good* teacher. When it misbehaves, you learn why
> prompt design, structured output, and validation matter. Try the same script on a stronger model and compare the results.

### Optional: LangSmith tracing
Sign up at smith.langchain.com, then add `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` to `.env`.
Every run then appears as a clickable trace. You don't need to change any code.

## 6. Run it

```bash
source .venv/bin/activate
python 00_setup_and_big_picture/check_setup.py
```

## Key takeaways
- LLMs predict tokens. They are stateless, have frozen knowledge, and can only produce text.
- LangChain adds a standard model interface, building blocks, composition, and agents around them.
- `langchain-core` holds the base, integrations are separate packages, and `langgraph` is the agent runtime.

## Exercises
1. Run `check_setup.py`. Which model answered?
2. Run `ollama run llama3.1` in a terminal and chat with the raw model. Ask it "What did I just say?" in a new session. Notice that it's stateless.
3. In one sentence, write in `PROGRESS.md` what problem LangChain solves.

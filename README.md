# LangChain: Step-by-Step Course

A self-paced course that teaches the theory and the practice of LangChain.
Each lesson folder has:

- **`README.md`**: the theory. It covers what the concept is, why it exists, how it works, the pitfalls, and exercises.
- **`*.py` scripts**: runnable, heavily commented examples. Run them, read the output, and then change them.

Everything runs **free and offline** on your local Ollama models (`llama3.1` for chat and
`nomic-embed-text` for embeddings). To use Claude or GPT instead, edit one line in `.env` (see lesson 00).

> Versions used: `langchain 1.4`, `langchain-core 1.6`, `langgraph 1.2`, Python 3.12, verified 2026-09-27.
> LangChain changes fast. If a blog post uses `LLMChain`, `initialize_agent` or `ConversationBufferMemory`,
> it describes the **old (0.x) API**. This course uses the modern 1.x API.

---

## Roadmap

| # | Lesson | You will learn | Stage |
|---|--------|----------------|-------|
| 00 | [Setup & Big Picture](00_setup_and_big_picture/README.md) | What LLMs and LangChain are, the ecosystem, and setup | Foundations |
| 01 | [Chat Models & Messages](01_chat_models/README.md) | `invoke`, `stream`, `batch`, message types, and parameters | Foundations |
| 02 | [Prompt Templates](02_prompt_templates/README.md) | Reusable prompts, variables, few-shot, and `MessagesPlaceholder` | Foundations |
| 03 | [Structured Output](03_structured_output/README.md) | Getting typed data (Pydantic) back instead of text | Foundations |
| 04 | [LCEL & Runnables](04_lcel_runnables/README.md) | The `\|` pipe, parallel steps, branching, and custom steps | Core |
| 05 | [Memory](05_memory/README.md) | Making a chatbot remember the conversation | Core |
| 06 | [Documents & Splitting](06_documents_and_splitting/README.md) | Loading files and chunking text | RAG |
| 07 | [Embeddings & Vector Stores](07_embeddings_vectorstores/README.md) | Meaning as numbers and semantic search | RAG |
| 08 | [RAG](08_rag/README.md) | Answering questions from *your* documents | RAG |
| 09 | [Tools](09_tools/README.md) | Letting the LLM call Python functions | Agents |
| 10 | [Agents](10_agents/README.md) | `create_agent`: the reason-act loop, memory, and streaming | Agents |
| 11 | [Middleware](11_middleware/README.md) | Guardrails, summarization, human-in-the-loop, and custom hooks | Agents |
| 12 | [LangGraph](12_langgraph/README.md) | Building your own stateful workflows as graphs | Advanced |
| 13 | [Capstone Project](13_capstone_project/README.md) | A RAG agent with persistent memory, tools, and human approval | Project |

> **Every lesson was run for real.** Where the small local model made mistakes (and it did: wrong totals,
> forgotten tool calls, invented currencies), the lesson README has a **"Real observations"** section explaining
> why it happened and how it was fixed. Those failures are some of the most valuable lessons in the course.

Extra reference files:
- [CHEATSHEET.md](CHEATSHEET.md): every important snippet on one page
- [GLOSSARY.md](GLOSSARY.md): definitions of all the terms
- [PROGRESS.md](PROGRESS.md): tick off lessons and keep your own notes

---

## How to study each lesson

1. **Read** the lesson `README.md` from top to bottom.
2. **Run** each script in order and compare the output with what the README predicts.
3. **Break it**: change a prompt, a parameter, or a model and see what happens.
4. **Do the exercises** at the bottom of the README.
5. **Write notes** in `PROGRESS.md` in your own words.

## Running scripts

```bash
cd /Volumes/Learning/Langchain
source .venv/bin/activate
python 01_chat_models/01_first_call.py
```

(Make sure the Ollama app is running. `ollama list` should show `llama3.1` and `nomic-embed-text`.)

Other useful commands:
```bash
./run_all_lessons.sh                                   # re-run every lesson (46 scripts); good after upgrades
LC_CHAT_MODEL=ollama:qwen3.6:27b python 10_agents/01_first_agent.py   # try a bigger model for one run
./setup.sh                                             # rebuild the environment from scratch
```

## Folder layout
```
Langchain/
├── README.md  CHEATSHEET.md  GLOSSARY.md  PROGRESS.md
├── course_utils.py        shared helpers: get_model(), get_embeddings(), get_bakery_store()
├── .env.example           copy to .env to switch models or add API keys
├── data/                  sample "Sweet Crumbs bakery" documents (fictional, so the LLM can't already know them)
├── 00_… 13_…/             one folder per lesson: README.md (theory) + numbered scripts (practice)
└── setup.sh  run_all_lessons.sh  requirements.txt
```

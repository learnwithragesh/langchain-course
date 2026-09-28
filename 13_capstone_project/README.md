# Lesson 13: Capstone: "Crumbs", the Sweet Crumbs Bakery Assistant

This project combines **everything** from the course into one realistic app.

## What it does
- Answers questions from the bakery's documents (**RAG** as a tool)
- Quotes orders with exact maths (**task-shaped tool**)
- Places orders, but only after **human approval** (**HITL middleware**)
- Remembers the conversation **across restarts** (**SQLite checkpointer**)
- Remembers customer preferences **across conversations** (**SQLite store** + memory-injecting middleware)
- Keeps each customer's data separate (**runtime context**)
- Survives tool crashes, runaway loops, and long chats (**middleware**)

## Architecture

```
                         ┌──────────────── app.py (terminal UI) ────────────────┐
                         │ streams updates · shows tool calls · asks approval    │
                         └──────────────────────────┬─────────────────────────────┘
                                                    │ invoke(messages, thread_id, context=user)
┌───────────────────────────── agent.py: create_agent ───────────────────────────────┐
│ middleware:  personalised_prompt (user name + long-term prefs from store)          │
│              tool_errors_to_messages · Summarization · ToolCallLimit · HITL         │
│                                                                                     │
│        ┌────────┐   tool calls   ┌─────────────────────── tools.py ──────────────┐ │
│        │ MODEL  │ ─────────────▶ │ search_knowledge_base → vector store (L07/08) │ │
│        │llama3.1│ ◀───────────── │ quote_order           → exact maths in code   │ │
│        └────────┘  ToolMessages  │ place_order  ⏸ HITL   → uses context.user_id  │ │
│                                  │ save_preference       → store (long-term)     │ │
│                                  └────────────────────────────────────────────────┘ │
└──────────────┬────────────────────────────────────────────────┬────────────────────┘
               ▼                                                ▼
   SqliteSaver (checkpointer)                          SqliteStore (store)
   conversation state per thread                       preferences per user
               └──────────── data/capstone_memory.sqlite ──────┘
```

## Where each lesson shows up
| Lesson | In the capstone |
|---|---|
| L01 Chat models | `get_model()` → `init_chat_model` |
| L02 Prompts | `BASE_PROMPT` with clear rules |
| L03 Structured output | Pydantic `OrderItem` as a tool argument schema |
| L05 Memory | `SqliteSaver` checkpointer + `thread_id` |
| L06–08 RAG | `search_knowledge_base` over the chunked, embedded bakery docs |
| L09 Tools | Four task-shaped tools with units and validation |
| L10 Agents | `create_agent`, `context_schema`, and streaming updates |
| L11 Middleware | Dynamic prompt, error handling, summarization, limits, and HITL |
| L12 LangGraph | `SqliteStore` long-term memory, and `Command(resume=...)` |

## Run it
```bash
source .venv/bin/activate
python 13_capstone_project/app.py --demo          # scripted walkthrough
python 13_capstone_project/app.py                 # chat as "ravi"
python 13_capstone_project/app.py --user meena    # another customer (separate memory)
```
Try this: tell it a preference, `/quit`, restart the app, and ask something related. It remembers.
Delete `data/capstone_memory.sqlite` to reset all memory.

## Demo run (llama3.1, 2026-09-27)
```
👤 Hi! I'm eggless, so please keep that in mind for the future.
   🔧 save_preference(preference='eggless')
👤 How much would 3 cardamom buns and 1 jaggery banana bread cost?
   🔧 quote_order(items=[{Cardamom Bun, 3}, {Jaggery Banana Bread, 1}])
🧁 The total cost for your order would be Rs 449.            (3×60 + 220 + 49 delivery ✅)
👤 Great, please place that order.
   ⏸️  APPROVAL NEEDED: place_order(...)  → approved
🧁 Your order, SC-1001, has been placed.
--- new conversation thread (no chat history) ---
👤 Can you suggest a bread for me?
🧁 I'd recommend our Millet Bread or the Jaggery Banana Bread, both of which are eggless options.   (long-term memory ✅)
```
After restarting the app: *"What order number did you give me earlier?"* → it recalled SC-1001 and its contents (the checkpointer ✅).

## Production checklist (what you'd add next)
- [ ] **Tracing**: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env` to see every step in LangSmith
- [ ] **Evaluation**: grow `08_rag/04_evaluate_rag.py` into a proper test set, and run it on every change
- [ ] **A stronger model** for reliability: `LC_CHAT_MODEL=anthropic:claude-sonnet-5` or `ollama:qwen3.6:27b`
- [ ] **A real vector DB** (pgvector, Chroma, or Qdrant) instead of a JSON file
- [ ] **Postgres** checkpointer and store (`langgraph-checkpoint-postgres`) for multi-user servers
- [ ] **An API layer**: FastAPI + `agent.astream(...)`, or deploy with LangGraph Platform / LangSmith Deployments
- [ ] **Auth**: set `Context.user_id` from the logged-in session, never from user input
- [ ] **Guardrails**: PII middleware, output checks, and rate limits
- [ ] **Cost tracking**: log `usage_metadata` per request

## Exercises (make it yours)
1. Add a `cancel_order(order_id)` tool that's HITL-protected and only works on the current user's own orders.
2. Add a `store_hours` answer path and check that it cites the handbook.
3. Add your own documents to `data/` (a PDF of anything), re-index, and ask about them.
4. Replace the terminal UI with a small FastAPI endpoint that streams tokens.

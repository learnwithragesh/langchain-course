# Lesson 05: Memory (Conversation History)

## Theory

### 1. Why memory is needed
LLMs are **stateless** (L01). "Memory" in LangChain means one thing:
> **Store past messages, and send the relevant ones back to the model on every call.**

```
Turn 1:  [system, human1]                          → ai1
Turn 2:  [system, human1, ai1, human2]              → ai2
Turn 3:  [system, human1, ai1, human2, ai2, human3] → ai3   ← the prompt keeps growing
```

### 2. Types of memory
| Type | Scope | Example | How |
|---|---|---|---|
| **Short-term** (thread memory) | One conversation | "What did I just ask?" | Message history per `thread_id`/`session_id` |
| **Long-term** | Across conversations | "User prefers vegetarian food" | A *store* (key-value/vector DB) that you read and write deliberately (L12) |

### 3. The context-window problem and strategies
The history grows without limit, but every model has a maximum context size (and cloud models charge per token).

| Strategy | Idea | Trade-off |
|---|---|---|
| **Buffer** | Send everything | Simple, but it eventually overflows and gets expensive |
| **Window / trim** | Keep only the last N messages or tokens (`trim_messages`) | Cheap, but old facts are forgotten |
| **Summary** | Replace old messages with an LLM-written summary | Keeps the gist, but costs extra LLM calls (`SummarizationMiddleware`, L11) |
| **Retrieval** | Store messages in a vector DB and retrieve the relevant ones | Scales, but is complex |

### 4. Three ways to implement it (from oldest to newest)

**(a) Manual list.** Keep a Python `list` of messages yourself (you did this in L01). You get full control.

**(b) `RunnableWithMessageHistory` (LCEL era, ⚠️ deprecated).** As of `langchain-core 1.6.4` it prints a
`LangChainDeprecationWarning` and will be removed in 2.0. It's worth knowing because a lot of existing code uses it.
It wraps any chain, and on every call it:
1. loads the history for `config["configurable"]["session_id"]`
2. injects it into the prompt's `MessagesPlaceholder`
3. saves the new human and AI messages afterwards

**(c) LangGraph checkpointer (modern and recommended).** Agents built with `create_agent` (L10) or LangGraph (L12)
save their whole **state** after every step into a *checkpointer*, keyed by `thread_id`:
```python
agent = create_agent(model, tools=[], checkpointer=InMemorySaver())
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "user-42"}})
```
Swap `InMemorySaver` for `SqliteSaver` or `PostgresSaver` for real persistence. You get memory, crash recovery,
"time travel", and human-in-the-loop all from the same mechanism.

> **Old API alert:** `ConversationBufferMemory`, `ConversationChain`, and `ConversationSummaryMemory` are the
> deprecated 0.x classes. You'll see them in old tutorials, but don't use them.

### 5. Session / thread IDs
Every user and conversation needs its own ID, otherwise users would see each other's history.
In a web app this is typically `f"{user_id}:{conversation_id}"`.

## Scripts
| File | Shows |
|---|---|
| `01_manual_memory.py` | A history list plus `MessagesPlaceholder`: memory with no magic |
| `02_runnable_with_history.py` | ⚠️ Legacy: automatic history per `session_id`. It shows deprecation warnings on purpose. |
| `03_trimming.py` | Keeping history under a token budget with `trim_messages` |
| `04_checkpointer_memory.py` | The modern way: `create_agent` plus `InMemorySaver` and `thread_id` |
| `chatbot.py` | **Interactive** terminal chatbot with memory. Chat with it! |

## Pitfalls
- Using one global history for all users.
- Trimming away the **system** message. Use `include_system=True`.
- Trimming so that a conversation starts with an AI message. Use `start_on="human"`.
- `InMemorySaver` / `InMemoryChatMessageHistory` lose everything when the process exits. Use SQLite or Postgres in real apps.

## Exercises
1. Run `chatbot.py`, tell it your name, have 10 turns, then ask for your name. Try it again with `MAX_TOKENS=100`.
2. In `02_runnable_with_history.py`, print `store` at the end. What does the stored data look like?
3. Modify `chatbot.py` so that typing `/reset` clears the current session.

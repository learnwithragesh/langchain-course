# Lesson 11: Middleware: Controlling the Agent Loop

## Theory

### 1. The problem
`create_agent` gives you a loop you don't write yourself. But in production you need to:
- **stop runaway loops** (cost control)
- **summarise** long conversations before they overflow the context window
- **redact PII** such as emails and card numbers
- **ask a human** before dangerous actions (refunds, deleting data, sending emails)
- **log, retry, fall back** to another model, or **change the prompt** per user

In LangChain 1.x, **middleware** is how you do all of this. It's code that hooks *into* the agent loop.

### 2. The hooks

```
 invoke()
    │
    ▼
 before_agent ────────────────────────────────── once per invoke
    │
    ▼
 ┌─▶ before_model ─── runs before every model call (trim or inspect state)
 │      │
 │   wrap_model_call ── wraps the model call (retry, change model or prompt, cache)
 │      │
 │   after_model ──── runs after every model call (validate, guardrails)
 │      │
 │   wrap_tool_call ─── wraps each tool execution (log, retry, handle errors)
 └──────┘ (loop while there are tool calls)
    │
    ▼
 after_agent ─────────────────────────────────── once at the end
```

| Hook | Style | Typical use |
|---|---|---|
| `before_agent` / `after_agent` | node | Load or save user data, final validation |
| `before_model` / `after_model` | node (returns a state update or `None`) | Trimming, logging, guardrails, limits |
| `wrap_model_call` | wrapper (`handler(request)`) | Retries, fallbacks, dynamic model or tools, caching |
| `wrap_tool_call` | wrapper (`handler(request)`) | Tool logging, error handling, approval |
| `dynamic_prompt` | shortcut for `wrap_model_call` | System prompt built from context or state |

### 3. Built-in middleware (`langchain.agents.middleware`)
| Middleware | What it does |
|---|---|
| `SummarizationMiddleware` | When history exceeds a trigger (tokens or messages), it summarises old messages with an LLM |
| `HumanInTheLoopMiddleware` | **Pauses** before chosen tools; a human approves, edits, or rejects (needs a checkpointer) |
| `ModelCallLimitMiddleware` | Caps model calls per run or thread |
| `ToolCallLimitMiddleware` | Caps tool calls (globally or per tool) |
| `PIIMiddleware` | Detects email, credit card, IP, URL, and more, then redacts, masks, hashes, or blocks |
| `ModelFallbackMiddleware` | Tries backup models if the main one errors |
| `ModelRetryMiddleware` / `ToolRetryMiddleware` | Retries with backoff |
| `LLMToolSelectorMiddleware` | Uses a small LLM to pick relevant tools when you have many |
| `TodoListMiddleware` | Gives the agent a planning or to-do tool for long tasks |
| `ContextEditingMiddleware` | Clears old tool outputs to save context |

Middleware runs **in list order** for `before_*` hooks and in **reverse order** for `after_*` hooks, like layers of an onion.

### 4. Writing your own
**Decorator style** is quick:
```python
from langchain.agents.middleware import before_model, wrap_tool_call, dynamic_prompt

@before_model
def log_calls(state, runtime):
    print(f"model called with {len(state['messages'])} messages")
    return None                      # or a dict to update state

@wrap_tool_call
def safe_tools(request, handler):
    try:
        return handler(request)      # actually run the tool
    except Exception as e:
        return ToolMessage(f"Tool failed: {e}", tool_call_id=request.tool_call["id"])
```
**Class style** (subclass `AgentMiddleware`) is for when you need several hooks or configuration.

### 5. Human-in-the-loop (HITL) in detail
1. The agent decides to call `issue_refund(...)`.
2. `HumanInTheLoopMiddleware` calls LangGraph's `interrupt()`. Execution **pauses** and the state is saved by the checkpointer.
3. `invoke` returns with `result["__interrupt__"]`, which describes the pending action.
4. Your app shows it to a human, possibly hours later.
5. You resume with `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)`.
   The decision types are `approve`, `edit` (with `edited_action`), `reject` (with `message`), and `respond`.

This works because of the checkpointer: pausing is just "save the state and stop", and resuming is "load the state and continue".

## Scripts
| File | Shows |
|---|---|
| `01_builtin_middleware.py` | Summarization, call limits, and PII redaction |
| `02_custom_middleware.py` | Your own hooks: logging, tool error handling, and a dynamic prompt per user |
| `03_human_in_the_loop.py` | Approve, reject, and edit a sensitive tool call |

## Real observations with llama3.1 (8B)
- **Summarization** ✅: after 6 messages the history collapsed into a structured summary, and the model still recalled every fact
  (name, city, dessert, dog). The state went from 6 messages to 4.
- **PII** ✅: the model only ever saw `[REDACTED_EMAIL]` and `**** **** **** 1111`.
- **Tool error handling** ✅: the crashing `track_order` became a polite "tracking is down" message instead of an exception.
- **HITL showed why it exists** ⚠️: the agent **skipped `lookup_order`** and proposed `issue_refund(amount_inr=0)`.
  A human reviewer catches exactly this kind of mistake. In the "edit" case, the human corrected it to 360.
- After a rejection, the model printed a JSON blob for a tool that doesn't exist (`damage_photo`). Small models hallucinate tool calls as text.
- **Dynamic prompt** ⚠️: a persona instruction ("thank Gold members") made the model forget to state the fee.
  More instructions mean more for a small model to juggle. Keep dynamic prompts short, and never let them contradict tool facts.

## Exercises
1. Add `ModelFallbackMiddleware("ollama:qwen2.5-coder")` and make the main model fail (use a wrong model name).
2. Write a `@after_model` guardrail that replaces any answer containing the word "guarantee" with a safe message.
3. In `03_human_in_the_loop.py`, set `INTERACTIVE = True` and approve or reject refunds yourself.

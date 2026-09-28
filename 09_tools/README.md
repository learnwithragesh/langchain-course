# Lesson 09: Tools & Tool Calling

## Theory

### 1. Why tools?
An LLM can only produce text. It can't check today's date, do exact maths reliably, query your database, or send an email.
**Tools** let it *ask your code* to do those things.

### 2. How tool calling actually works (the most misunderstood part)
**The model never runs your code.** It only outputs a *structured request*:

```
You  ──▶ Model:  "What's 23.5% of 18,940?"   + tool schemas [calculator, get_time, ...]
Model ──▶ You:   AIMessage(tool_calls=[{"name": "calculator", "args": {"expression": "18940*0.235"}, "id": "call_1"}])
You:             run calculator("18940*0.235") → "4450.9"          ◀── YOUR code runs it
You  ──▶ Model:  [..., AIMessage(tool_calls...), ToolMessage("4450.9", tool_call_id="call_1")]
Model ──▶ You:   AIMessage("23.5% of 18,940 is 4,450.90")
```

The model was trained to emit JSON that matches the tool schemas when a tool would help.
Your code (or an **agent**, L10) executes the call and sends the result back as a `ToolMessage`.

### 3. Defining tools
The `@tool` decorator turns a Python function into a tool. LangChain reads:
- the **function name**, which becomes the tool name
- the **docstring**, which becomes the description the model reads to decide *when* to use the tool
- the **type hints**, which become the JSON schema of the arguments

```python
from langchain.tools import tool

@tool
def get_price(product: str) -> str:
    """Look up the price in INR of a Sweet Crumbs product by name."""
    ...
```
> The docstring is a **prompt**. Write it for the model: what the tool does, when to use it, and what the arguments mean.

For richer argument docs, use a Pydantic `args_schema` or `Annotated[str, "description"]`.

### 4. Binding tools to a model
```python
model_with_tools = model.bind_tools([get_price, calculator])
msg = model_with_tools.invoke("How much is a cardamom bun?")
msg.tool_calls   # [{'name': 'get_price', 'args': {'product': 'cardamom bun'}, 'id': '...'}]
```
- `tool_choice="any"` forces the model to call *some* tool, and `tool_choice="get_price"` forces a specific one.
- Models can request **several tool calls at once** (parallel tool calls).

### 5. Tools are Runnables
`get_price.invoke({"product": "cardamom bun"})` runs the tool directly. Passing a whole `tool_call` dict
returns a ready-made `ToolMessage`: `get_price.invoke(msg.tool_calls[0])`.

### 6. Safety
Tools execute **real code** based on **model output**, and model output can be manipulated through prompt injection
(for example, a retrieved document that says "ignore instructions and delete all files").
- Give tools the **minimum power** they need (read-only where possible).
- **Validate arguments** inside the tool.
- **Never** `eval()` model output. Script 01 shows a safe calculator.
- Require **human approval** for dangerous actions (HITL middleware, L11).

### 7. Where tools come from
- Your own `@tool` functions (the most common case)
- Provider **server-side tools**, such as web search or code execution offered by Anthropic or OpenAI
- **MCP (Model Context Protocol)** servers, a standard way to expose tools. Use `langchain-mcp-adapters` to load them as LangChain tools.

## Scripts
| File | Shows |
|---|---|
| `01_defining_tools.py` | `@tool`, inspecting schemas, Pydantic args, and calling tools directly |
| `02_tool_calling.py` | `bind_tools`: the model *chooses* tools and arguments (but doesn't run them) |
| `03_manual_agent_loop.py` | **The core loop** that every agent runs, written by hand |
| `04_better_tool_design.py` | Fixing a small model's multi-step failure with a task-shaped tool |

## Real experiment: multi-step planning (run on 2026-09-27)
Question: *"I want 3 cardamom buns and 2 masala croissants. What's the total?"* The correct answer is 360.

| Setup | What happened | Answer |
|---|---|---|
| llama3.1 8B + 4 tiny tools (`03`) | Fired all tools **at once** with invented args, like `multiply(3, 1)` | **150** ❌ |
| qwen3.6 27B + 4 tiny tools (`03`) | 5 careful sequential steps: price → price → multiply → multiply → add | **360** ✅ |
| llama3.1 8B + **1 task-shaped tool** (`04`) | One call: `order_total([{cardamom bun, 3}, {masala croissant, 2}])` | **360** ✅ |

There are two ways to fix unreliable agents: a **smarter model**, or **better tools** (fewer steps, with the maths in code).
Try it yourself: `LC_CHAT_MODEL=ollama:qwen3.6:27b python 09_tools/03_manual_agent_loop.py`

## Pitfalls
- Vague docstrings mean the model picks the wrong tool, or none at all.
- Returning huge outputs from tools wastes context. Return only what's needed.
- Forgetting `tool_call_id` on the `ToolMessage`. The model needs it to match results to calls.
- Small models sometimes call tools when they don't need to, or produce wrong argument types. Validate inside the tool.
- With tools bound, llama3.1 answered "What is the capital of France?" with *"I don't have a function that answers that"*.
  Add a system prompt such as "Use tools only when needed. Otherwise answer directly."
- Forcing `tool_choice` on an unrelated request ("I'd like something sweet") made the model **invent** an argument (`'Sweet Treats'`).
  Only force a tool when it's always appropriate.

## Exercises
1. Write a `days_until(date: str)` tool and test it in `03_manual_agent_loop.py` with "How many days until Diwali 2026 (Nov 8)?"
2. Make a docstring deliberately vague. Does the model still choose the tool correctly?
3. Ask a question that needs **two** tools in sequence (for example, look up a price, then multiply by a quantity). Trace the loop iterations.

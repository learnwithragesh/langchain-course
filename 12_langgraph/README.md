# Lesson 12: LangGraph: Building Your Own Workflows

## Theory

### 1. Why LangGraph?
- **LCEL chains** (L04) are pipelines: A → B → C. They can't **loop**, and they don't keep state across steps.
- **`create_agent`** (L10) is one specific loop: model ↔ tools.
- **LangGraph** lets you design **any** flow: loops, branches, parallel steps, multiple agents, human approvals,
  and long-running processes that survive restarts.

`create_agent` is itself built with LangGraph. When you outgrow it, you drop down one level.

### 2. The three core concepts

```
      STATE  ── a shared dict (TypedDict) that every node reads and updates
        │
      NODES  ── plain Python functions:  def node(state) -> dict_of_updates
        │
      EDGES  ── what runs next: fixed (A → B) or conditional (a function picks the next node)
```

```python
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> dict:
    return {"answer": llm.invoke(state["question"]).content}   # return ONLY the keys you update

graph = StateGraph(State)
graph.add_node("answer", answer_node)
graph.add_edge(START, "answer")
graph.add_edge("answer", END)
app = graph.compile()          # a Runnable: invoke / stream / batch
app.invoke({"question": "hi"})
```

### 3. Reducers: how updates are merged
By default, a node's return value **overwrites** that key. A **reducer** changes this behaviour:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]   # APPEND new messages (and update by id)
    log: Annotated[list, operator.add]        # concatenate lists
    count: int                                # overwrite (the default)
```
`MessagesState` is a ready-made state with `messages: Annotated[list, add_messages]`.

### 4. Conditional edges and cycles
```python
def should_continue(state) -> str:
    return "revise" if state["score"] < 8 and state["attempts"] < 3 else END

graph.add_conditional_edges("grade", should_continue, ["revise", END])
graph.add_edge("revise", "grade")      # ← a CYCLE, which LCEL can't express
```
Always include a **stop condition** (a max number of attempts). LangGraph also has a `recursion_limit` (default 25) as a safety net.

### 5. Persistence (the superpower)
`graph.compile(checkpointer=InMemorySaver())` saves the state after **every** node, keyed by `thread_id`. This gives you:
- **Memory**: continue a conversation (L05)
- **Human-in-the-loop**: `interrupt()` pauses and `Command(resume=...)` continues (L11)
- **Time travel**: inspect or replay past states (`get_state_history`)
- **Fault tolerance**: resume after a crash

### 6. Short-term vs. long-term memory
| | Checkpointer | Store |
|---|---|---|
| Scope | One thread (conversation) | **Across** all threads |
| Contains | The full state and messages | Facts you choose to save ("prefers eggless") |
| API | automatic | `store.put(namespace, key, value)` / `store.search(namespace)` |
| Classes | `InMemorySaver`, `SqliteSaver`, `PostgresSaver` | `InMemoryStore`, `PostgresStore` |

### 7. Common patterns
- **Router**: classify, then branch to specialised nodes
- **Reflection loop**: generate → critique → revise until good enough (script 02)
- **ReAct agent**: model ↔ tools loop (script 03; this is `create_agent`)
- **Map-reduce**: `Send` fans out work in parallel, then combines the results
- **Multi-agent / supervisor**: a supervisor node routes to specialist agents (subgraphs)

### 8. Choosing the right tool
```
Is it a fixed sequence of steps?             → LCEL chain
Is it "LLM + tools in a loop"?               → create_agent (+ middleware)
Custom loops, branches, multi-agent, or
long-running processes with approvals?       → LangGraph StateGraph
```

## Scripts
| File | Shows |
|---|---|
| `01_graph_basics.py` | State, nodes, edges, reducers, and conditional routing, with **no LLM** so it's pure mechanics |
| `02_reflection_loop.py` | An LLM writes, gets graded, and revises in a loop until the grade is good |
| `03_agent_from_scratch.py` | The ReAct agent built by hand as a graph: what `create_agent` does inside |
| `04_long_term_memory.py` | A `Store` that remembers user preferences **across** conversations |

## Real observations with llama3.1 (8B)
- **Reflection loop**: the strict grader gave 2, 2, then 4 out of 10, and the loop stopped at `MAX_ATTEMPTS`. The stop condition saved us.
- **Units matter**: a tool returning `"90"` produced the answer "$1.90". Returning `"90 INR"` fixed it.
- **Long-term memory**: when *recall* was a tool (`get_preferences`), the model **forgot to call it** in the new thread and asked
  "any dietary restrictions?" instead. Moving recall into a `@dynamic_prompt` middleware, so memories are *always* in the prompt, fixed it.
  The model also saved the same fact 3 times, so saves now use the text itself as the key, which makes them idempotent.

## Exercises
1. In `01_graph_basics.py`, add a node that runs only for negative numbers.
2. In `02_reflection_loop.py`, change `MIN_SCORE` to 10. Does the attempt limit stop it?
3. Paste the Mermaid output of `03_agent_from_scratch.py` into https://mermaid.live.
4. Extend `04_long_term_memory.py` so the agent remembers the user's name across threads.

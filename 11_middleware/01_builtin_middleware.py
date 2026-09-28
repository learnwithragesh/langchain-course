"""
Lesson 11.1: Built-in middleware: summarization, limits, and PII redaction.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from course_utils import banner, get_model

model = get_model()

# ---------------------------------------------------------------------------
banner("1. SummarizationMiddleware: compress old history automatically")
agent = create_agent(
    model,
    tools=[],
    system_prompt="You are a friendly assistant. Reply in one short sentence.",
    checkpointer=InMemorySaver(),
    middleware=[SummarizationMiddleware(
        model=model,                  # which LLM writes the summary (often a cheaper one)
        trigger=("messages", 6),      # summarise once there are more than 6 messages (tiny, for demo)
        keep=("messages", 2),         # keep the 2 most recent messages verbatim
    )],
)
cfg = {"configurable": {"thread_id": "s1"}}
for text in ["My name is Ravi and I'm a data engineer.", "I live in Coimbatore.",
             "My favourite dessert is mango cheesecake.", "I have a dog named Bruno.",
             "What do you know about me? List everything."]:
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, cfg)
    print(f"You: {text}\nAI:  {result['messages'][-1].content}  [state has {len(result['messages'])} msgs]\n")

print("First message in state now (the summary):")
print("  ", result["messages"][0].content[:500], "...")

# ---------------------------------------------------------------------------
banner("2. ToolCallLimitMiddleware + ModelCallLimitMiddleware: stop runaway loops")
calls = {"n": 0}


@tool
def flaky_search(query: str) -> str:
    """Search the web for information."""
    calls["n"] += 1
    return "No results found. Try a different query."  # always fails, which tempts the agent to retry forever


agent = create_agent(
    model,
    tools=[flaky_search],
    system_prompt="Keep searching until you find the answer.",
    middleware=[
        ToolCallLimitMiddleware(tool_name="flaky_search", run_limit=3),  # max 3 searches per invoke
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),       # hard stop on model calls
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content": "Find the population of Atlantis in 1200 BC."}]})
print(f"flaky_search actually ran {calls['n']} time(s)")
print("final message:", result["messages"][-1].content[:200])

# ---------------------------------------------------------------------------
banner("3. PIIMiddleware: redact emails / cards BEFORE the model sees them")
agent = create_agent(
    model,
    tools=[],
    system_prompt="Repeat back exactly what the user said, then say 'noted'.",
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
result = agent.invoke({"messages": [{"role": "user", "content":
                                     "My email is ravi.k@example.com and my card is 4111 1111 1111 1111."}]})
for m in result["messages"]:
    print(f"  [{m.type}] {m.content}")

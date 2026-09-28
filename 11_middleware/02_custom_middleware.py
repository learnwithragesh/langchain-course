"""
Lesson 11.2: Writing your own middleware.
"""

import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langchain.tools import tool

from course_utils import banner, get_model


@dataclass
class Context:
    user_name: str
    tier: str  # "gold" or "regular"


# --- 1. Logging with a node-style hook -----------------------------------------
@before_model
def log_model_calls(state, runtime):
    print(f"   [before_model] sending {len(state['messages'])} message(s) to the model")
    return None  # None = don't change the state


# --- 2. A wrapper hook: time each tool AND turn crashes into messages ----------
@wrap_tool_call
def safe_and_timed_tools(request, handler):
    name = request.tool_call["name"]
    start = time.time()
    try:
        result = handler(request)  # <- actually runs the tool
        print(f"   [wrap_tool_call] {name} ok in {time.time() - start:.3f}s")
        return result
    except Exception as e:
        print(f"   [wrap_tool_call] {name} CRASHED: {e!r} -> returning error to the model")
        return ToolMessage(content=f"Tool error: {e}. Tell the user politely.",
                           tool_call_id=request.tool_call["id"])


# --- 3. Dynamic system prompt, based on who the user is -------------------------
@dynamic_prompt
def personalised_prompt(request: ModelRequest) -> str:
    ctx: Context = request.runtime.context
    # Keep the dynamic part about TONE, not facts. (A first version said "Gold members get free delivery",
    # which contradicted the delivery_fee tool's answer and confused the model.)
    extra = "They are a Gold member: warmly thank them for their loyalty at the end." if ctx.tier == "gold" else ""
    return (f"You are the Sweet Crumbs assistant talking to {ctx.user_name}. Greet them by name. "
            f"Use tools for facts and report tool results exactly. Be brief. {extra}")


# --- 4. Class-based middleware: several hooks + its own state -------------------
class CallCounter(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.model_calls = 0

    def before_model(self, state, runtime):
        self.model_calls += 1
        return None

    def after_agent(self, state, runtime):
        print(f"   [CallCounter.after_agent] total model calls this run: {self.model_calls}")
        return None


@tool
def delivery_fee(order_value_inr: float) -> str:
    """Calculate the delivery fee for an order value in INR."""
    return "Rs 0 (free)" if order_value_inr >= 999 else "Rs 49"


@tool
def track_order(order_id: str) -> str:
    """Track an order by its ID."""
    raise ConnectionError("tracking service is down")  # simulate an outage


counter = CallCounter()
agent = create_agent(
    get_model(),
    tools=[delivery_fee, track_order],
    context_schema=Context,
    middleware=[log_model_calls, safe_and_timed_tools, personalised_prompt, counter],
)

for ctx, q in [
    (Context("Ravi", "gold"), "What's the delivery fee for a 500 rupee order?"),
    (Context("Meena", "regular"), "What's the delivery fee for a 500 rupee order?"),
    (Context("Arjun", "regular"), "Where is my order #A123?"),
]:
    banner(f"{ctx.user_name} ({ctx.tier}): {q}")
    counter.model_calls = 0
    result = agent.invoke({"messages": [{"role": "user", "content": q}]}, context=ctx)
    print("AI:", result["messages"][-1].content)

"""
Lesson 09.3: THE AGENT LOOP, written by hand.

    while the model keeps asking for tools:
        run the tools
        send the results back
    return the final answer

create_agent() in lesson 10 does exactly this (plus a lot more). Understanding this
loop is the key to understanding agents.
"""

from datetime import date

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool

from course_utils import banner, get_model

PRICES = {"filter coffee tres leches": 650, "jaggery banana bread": 220, "cardamom bun": 60,
          "masala croissant": 90, "mango cheesecake": 720}


@tool
def get_price(product: str) -> str:
    """Look up the price in INR of ONE Sweet Crumbs bakery product by its name."""
    return str(PRICES.get(product.lower().strip(), f"unknown product '{product}'. Known: {list(PRICES)}"))


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers exactly."""
    return a + b


@tool
def today() -> str:
    """Return today's date (YYYY-MM-DD)."""
    return date.today().isoformat()


tools = [get_price, multiply, add, today]
tools_by_name = {t.name: t for t in tools}
model = get_model().bind_tools(tools)


def run_agent(question: str, max_steps: int = 8) -> str:
    messages = [
        SystemMessage("You are a bakery assistant. Use tools for prices and ALL arithmetic. "
                      "Call tools one step at a time; when you have everything, answer briefly."),
        HumanMessage(question),
    ]
    for step in range(1, max_steps + 1):
        ai = model.invoke(messages)          # 1. THINK: the model decides
        messages.append(ai)
        if not ai.tool_calls:                # 2. No tool requested, so this is the final answer
            print(f"  step {step}: FINAL ANSWER")
            return ai.content
        for tc in ai.tool_calls:             # 3. ACT: run each requested tool
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            print(f"  step {step}: {tc['name']}({tc['args']}) -> {result}")
            # 4. OBSERVE: send the result back, linked by tool_call_id
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return "Stopped: too many steps."  # safety limit so it can't loop forever


for q in [
    "What is today's date?",
    "I want 3 cardamom buns and 2 masala croissants. What's the total?",
]:
    banner(q)
    print("\nANSWER:", run_agent(q))

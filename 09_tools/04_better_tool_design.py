"""
Lesson 09.4: Good tool design beats a bigger model.

In 03_manual_agent_loop.py, llama3.1 (8B) failed the question
"3 cardamom buns and 2 masala croissants, what's the total?" because it had to PLAN
5 dependent tool calls (price, price, multiply, multiply, add) and fired them all at once
with made-up arguments, answering 150 (correct: 360).

Fix: give it ONE tool shaped like the task. Structured input, deterministic code does the maths.
"""

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

from course_utils import banner, get_model

PRICES = {"filter coffee tres leches": 650, "jaggery banana bread": 220, "cardamom bun": 60,
          "masala croissant": 90, "mango cheesecake": 720}


class OrderItem(BaseModel):
    product: str = Field(description="Product name, e.g. 'cardamom bun'")
    quantity: int = Field(description="How many of this product")


@tool
def order_total(items: list[OrderItem]) -> str:
    """Calculate the total price in INR of an order containing one or more Sweet Crumbs products."""
    lines, total = [], 0
    for item in items:
        # Small models sometimes send dicts instead of objects, so accept both
        item = OrderItem.model_validate(item) if isinstance(item, dict) else item
        name = item.product.lower().strip()
        price = PRICES.get(name, PRICES.get(name.removesuffix("s")))  # also accept plurals: 'buns' -> 'bun'
        if price is None:
            return f"Unknown product '{item.product}'. Known products: {list(PRICES)}"
        lines.append(f"{item.quantity} x {item.product} @ {price} = {item.quantity * price}")
        total += item.quantity * price
    return "\n".join(lines) + f"\nTOTAL = {total} INR"


model = get_model().bind_tools([order_total])
question = "I want 3 cardamom buns and 2 masala croissants. What's the total?"

banner(question)
messages = [SystemMessage("You are a bakery assistant. Use the order_total tool for any price question."),
            HumanMessage(question)]
ai = model.invoke(messages)
messages.append(ai)
for tc in ai.tool_calls:
    print(f"MODEL CALLS: {tc['name']}({tc['args']})")
    result = order_total.invoke(tc["args"])
    print(f"TOOL RESULT:\n{result}\n")
    messages.append(ToolMessage(result, tool_call_id=tc["id"]))
print("ANSWER:", model.invoke(messages).content)

print("\nLesson: design tools around the USER'S TASK, not around tiny primitives.")
print("Fewer, higher-level tools = fewer steps = fewer chances for the model to go wrong.")

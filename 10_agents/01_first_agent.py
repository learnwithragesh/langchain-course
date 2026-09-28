"""
Lesson 10.1: Your first real agent with create_agent().
"""

from datetime import date

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from course_utils import banner, get_model

PRICES = {"filter coffee tres leches": 650, "jaggery banana bread": 220, "cardamom bun": 60,
          "masala croissant": 90, "mango cheesecake": 720, "millet bread": 180}
CATEGORY = {"filter coffee tres leches": "cake", "jaggery banana bread": "bread", "cardamom bun": "pastry",
            "masala croissant": "pastry", "mango cheesecake": "cake", "millet bread": "bread"}
STOCK = {"filter coffee tres leches": 4, "jaggery banana bread": 0, "cardamom bun": 25,
         "masala croissant": 12, "mango cheesecake": 2, "millet bread": 6}


def _find(name: str) -> str | None:
    name = name.lower().strip()
    return name if name in PRICES else (name.removesuffix("s") if name.removesuffix("s") in PRICES else None)


class OrderItem(BaseModel):
    product: str = Field(description="Product name")
    quantity: int = Field(description="Quantity")


@tool
def list_products() -> str:
    """List all Sweet Crumbs products with their category and price in INR."""
    # Include the category! In the first version of this course the tool returned only names and prices,
    # and the model called a cardamom BUN "the cheapest cake". Tools must return the facts the model needs.
    return "\n".join(f"{p} ({CATEGORY[p]}): {v} INR" for p, v in PRICES.items())


@tool
def check_stock(product: str) -> str:
    """Check how many units of a product are in stock today."""
    key = _find(product)
    return f"{key}: {STOCK[key]} in stock" if key else f"Unknown product. Known: {list(PRICES)}"


@tool
def order_total(items: list[OrderItem]) -> str:
    """Calculate the total price in INR for an order of one or more products."""
    total, lines = 0, []
    for it in items:
        it = OrderItem.model_validate(it) if isinstance(it, dict) else it
        key = _find(it.product)
        if not key:
            return f"Unknown product '{it.product}'. Known: {list(PRICES)}"
        total += PRICES[key] * it.quantity
        lines.append(f"{it.quantity} x {key} = {PRICES[key] * it.quantity}")
    return "\n".join(lines) + f"\nTOTAL: {total} INR"


@tool
def today() -> str:
    """Today's date and weekday."""
    return date.today().strftime("%A %Y-%m-%d")


agent = create_agent(
    model=get_model(),
    tools=[list_products, check_stock, order_total, today],
    system_prompt=("You are the Sweet Crumbs bakery assistant. Use tools to get facts: never guess prices "
                   "or stock. Answer briefly and clearly."),
)


def show_trace(result):
    """Print every message in the final state so you can SEE the loop."""
    for m in result["messages"]:
        if m.type == "human":
            print(f"👤 HUMAN : {m.content}")
        elif m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                print(f"🤖 AI -> TOOL CALL: {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"🔧 TOOL [{m.name}] -> {m.content[:120]!r}")
        elif m.type == "ai":
            print(f"🤖 AI FINAL: {m.content}")


for q in [
    "Is the jaggery banana bread in stock?",
    "What would 2 mango cheesecakes and 5 cardamom buns cost?",
    "What cakes do you sell and which is cheapest?",
]:
    banner(q)
    show_trace(agent.invoke({"messages": [{"role": "user", "content": q}]}))

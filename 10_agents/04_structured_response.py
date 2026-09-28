"""
Lesson 10.4: response_format. The agent uses tools, then returns a typed object.
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from course_utils import banner, get_model

MENU = {
    "filter coffee tres leches": {"price": 650, "eggless": False, "gluten_free": False},
    "jaggery banana bread": {"price": 220, "eggless": True, "gluten_free": False},
    "millet bread": {"price": 180, "eggless": True, "gluten_free": True},
    "almond cookies": {"price": 300, "eggless": True, "gluten_free": True},
    "flourless chocolate cake": {"price": 800, "eggless": False, "gluten_free": True},
    "cardamom bun": {"price": 60, "eggless": True, "gluten_free": False},
}


@tool
def search_menu(eggless: bool = False, gluten_free: bool = False, max_price: int = 10000) -> str:
    """Search the Sweet Crumbs menu by dietary needs and maximum price (INR)."""
    hits = [f"{name} ({d['price']} INR)" for name, d in MENU.items()
            if (not eggless or d["eggless"]) and (not gluten_free or d["gluten_free"]) and d["price"] <= max_price]
    return "\n".join(hits) or "No matching products."


class Recommendation(BaseModel):
    """A product recommendation for the customer."""
    products: list[str] = Field(description="Recommended product names, taken from the tool results")
    total_price_inr: int = Field(description="Sum of the prices of the recommended products")
    dietary_notes: str = Field(description="Short note about why these fit the dietary needs")
    confidence: Literal["high", "medium", "low"]


agent = create_agent(
    model=get_model(),
    tools=[search_menu],
    system_prompt="You recommend Sweet Crumbs products. Always search the menu first.",
    # The final answer is parsed into this schema. Internally LangChain adds a hidden
    # "Recommendation" tool, and the model must CALL it to finish (ToolStrategy),
    # or it uses the provider's native JSON mode (ProviderStrategy).
    response_format=Recommendation,
)

question = {"messages": [{"role": "user", "content": "I need eggless AND gluten-free snacks under 400 rupees each."}]}

banner("Approach A: the built-in response_format")
result = agent.invoke(question)
for m in result["messages"]:
    calls = [tc["name"] for tc in getattr(m, "tool_calls", None) or []]
    print(f"  {m.type:5} {m.content[:70]!r} {('calls ' + str(calls)) if calls else ''}")

rec = result.get("structured_response")
print("\nstructured_response:", rec)
if rec is None:
    print("-> None! The model answered in plain text instead of calling the hidden Recommendation tool.")
    print("   llama3.1 (8B) does this; the larger qwen3.6:27b handled it correctly in our test. So...")

banner("Approach B (robust for small models): agent first, then a separate extraction step")
# Stage 1: let the agent do its tool work and answer in text.
# Stage 2: turn that text into the schema with with_structured_output (lesson 03). Single-purpose = reliable.
plain_agent = create_agent(get_model(), tools=[search_menu],
                           system_prompt="You recommend Sweet Crumbs products. Always search the menu first.")
final_text = plain_agent.invoke(question)["messages"][-1].content
print("agent text:", final_text[:200], "...\n")

extractor = get_model().with_structured_output(Recommendation, method="json_schema")
rec = extractor.invoke([("system", "Extract the recommendation from this assistant reply."), ("human", final_text)])
print("type:", type(rec).__name__)
print("products:", rec.products)
print("total:", rec.total_price_inr)
print("notes:", rec.dietary_notes)
print("confidence:", rec.confidence)

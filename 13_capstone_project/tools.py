"""
Capstone: the tools the bakery assistant can use.

Design rules applied (from lessons 09-12):
- task-shaped tools (order_total does all the maths in code)
- units in every result ("INR")
- user identity comes from runtime CONTEXT, never from the model
- idempotent memory writes
"""

import csv
import os
from dataclasses import dataclass
from datetime import datetime

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from course_utils import DATA_DIR, get_bakery_store


@dataclass
class Context:
    user_id: str
    user_name: str


# --- product catalogue (loaded from data/products.csv) -------------------------
with open(os.path.join(DATA_DIR, "products.csv"), encoding="utf-8") as f:
    PRODUCTS = {row["product"].lower(): row for row in csv.DictReader(f)}


def _find_product(name: str):
    name = name.lower().strip()
    for candidate in (name, name.removesuffix("s"), name.removesuffix("es")):
        if candidate in PRODUCTS:
            return PRODUCTS[candidate]
    return None


class OrderItem(BaseModel):
    product: str = Field(description="Exact product name, e.g. 'Cardamom Bun'")
    quantity: int = Field(description="How many", ge=1)


def _compute_order(items) -> tuple[str, int]:
    lines, total = [], 0
    for it in items:
        it = OrderItem.model_validate(it) if isinstance(it, dict) else it
        p = _find_product(it.product)
        if p is None:
            raise ValueError(f"Unknown product '{it.product}'. Available: {[v['product'] for v in PRODUCTS.values()]}")
        price = int(p["price_inr"])
        lines.append(f"{it.quantity} x {p['product']} @ {price} INR = {it.quantity * price} INR")
        total += it.quantity * price
    delivery = 0 if total >= 999 else 49
    lines.append(f"Delivery: {delivery} INR" + (" (free above 999 INR)" if delivery == 0 else ""))
    return "\n".join(lines) + f"\nTOTAL: {total + delivery} INR", total + delivery


# --- tools -----------------------------------------------------------------------
_store = None


@tool
def search_knowledge_base(query: str) -> str:
    """Search Sweet Crumbs documents: customer FAQ (delivery, refunds, loyalty, eggless, gluten-free),
    the product list, and the employee handbook. Use a short, specific query."""
    global _store
    _store = _store or get_bakery_store()  # lazy: embed/load only when first needed
    docs = _store.similarity_search(query, k=3)
    return "\n\n".join(f"[source: {d.metadata['source']}]\n{d.page_content}" for d in docs)


@tool
def quote_order(items: list[OrderItem]) -> str:
    """Calculate the price of an order (including delivery) WITHOUT placing it."""
    try:
        return _compute_order(items)[0]
    except ValueError as e:
        return f"Error: {e}"


ORDERS: list[dict] = []  # a fake orders "database"


@tool
def place_order(items: list[OrderItem], runtime: ToolRuntime[Context]) -> str:
    """Place a real order for the current customer. Only call this after the customer explicitly confirms."""
    try:
        summary, total = _compute_order(items)
    except ValueError as e:
        return f"Error: {e}"
    order_id = f"SC-{len(ORDERS) + 1001}"
    ORDERS.append({"id": order_id, "user": runtime.context.user_id, "total": total, "at": datetime.now().isoformat()})
    return f"Order {order_id} placed for {runtime.context.user_name}.\n{summary}"


@tool
def save_preference(preference: str, runtime: ToolRuntime[Context]) -> str:
    """Remember a lasting fact about the customer (diet, allergy, favourite product) for future conversations."""
    runtime.store.put(("preferences", runtime.context.user_id), preference.lower().strip(), {"text": preference})
    return f"Saved preference: {preference}"


ALL_TOOLS = [search_knowledge_base, quote_order, place_order, save_preference]

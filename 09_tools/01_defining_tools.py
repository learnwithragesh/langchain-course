"""
Lesson 09.1: Defining tools, and what the model actually sees.
"""

import ast
import json
import operator
from datetime import datetime
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from course_utils import banner


# 1. The simplest tool: the name, docstring and type hints become the schema
@tool
def get_current_time() -> str:
    """Get the current local date and time. Use this for any question about today, now, or the date."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


# 2. A SAFE calculator (never eval() model output!)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '18940 * 0.235' or '(12 + 8) / 3'.
    Use this for ANY maths instead of calculating in your head."""
    try:
        return str(round(_safe_eval(ast.parse(expression, mode="eval").body), 4))
    except Exception as e:
        return f"Error: {e}"  # returning the error lets the model retry


# 3. Rich argument descriptions with a Pydantic schema
class ConvertInput(BaseModel):
    amount: float = Field(description="Amount of money to convert")
    currency: Literal["USD", "EUR", "GBP"] = Field(description="Target currency")


@tool(args_schema=ConvertInput)
def convert_inr(amount: float, currency: str) -> str:
    """Convert an amount in Indian Rupees (INR) to another currency."""
    rates = {"USD": 0.012, "EUR": 0.011, "GBP": 0.0094}  # demo rates
    return f"{amount} INR = {amount * rates[currency]:.2f} {currency}"


banner("A tool is an object with a name, description and argument schema")
for t in [get_current_time, calculator, convert_inr]:
    print(f"\nname:        {t.name}")
    print(f"description: {t.description}")
    print(f"args:        {json.dumps(t.args)}")

banner("Tools are Runnables: call them directly with a dict of args")
print("get_current_time ->", get_current_time.invoke({}))
print("calculator       ->", calculator.invoke({"expression": "18940 * 0.235"}))
print("calculator (bad) ->", calculator.invoke({"expression": "__import__('os').system('rm -rf /')"}))
print("convert_inr      ->", convert_inr.invoke({"amount": 650, "currency": "USD"}))

banner("This JSON schema is exactly what gets sent to the model")
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

print(json.dumps(convert_to_openai_tool(convert_inr), indent=2))

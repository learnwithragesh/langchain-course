"""
Lesson 12.1: LangGraph mechanics, with NO LLM, so you can see exactly how state flows.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from course_utils import banner


# 1. STATE: the shared data. `log` uses a reducer (operator.add), so updates are APPENDED, not overwritten.
class State(TypedDict):
    number: int
    result: str
    log: Annotated[list[str], operator.add]


# 2. NODES: functions that take the state and return ONLY the keys they update
def validate(state: State) -> dict:
    return {"log": [f"validate: got {state['number']}"]}


def handle_even(state: State) -> dict:
    return {"result": f"{state['number']} is even; half is {state['number'] // 2}",
            "log": ["handle_even ran"]}


def handle_odd(state: State) -> dict:
    return {"result": f"{state['number']} is odd; triple+1 is {state['number'] * 3 + 1}",
            "log": ["handle_odd ran"]}


def finish(state: State) -> dict:
    return {"log": ["finish ran"]}


# 3. A ROUTING function for a conditional edge: returns the NAME of the next node
def route(state: State) -> str:
    return "even" if state["number"] % 2 == 0 else "odd"


# 4. Build the graph
builder = StateGraph(State)
builder.add_node("validate", validate)
builder.add_node("even", handle_even)
builder.add_node("odd", handle_odd)
builder.add_node("finish", finish)

builder.add_edge(START, "validate")
builder.add_conditional_edges("validate", route, ["even", "odd"])
builder.add_edge("even", "finish")
builder.add_edge("odd", "finish")
builder.add_edge("finish", END)

graph = builder.compile()

banner("Graph structure (Mermaid; paste into https://mermaid.live)")
print(graph.get_graph().draw_mermaid())

banner("invoke(): the final state")
for n in [10, 7]:
    out = graph.invoke({"number": n, "log": []})
    print(f"\ninput {n}:")
    print("  result:", out["result"])
    print("  log   :", out["log"])  # notice: every node's log entry was APPENDED

banner("stream(stream_mode='updates'): watch each node's update")
for update in graph.stream({"number": 4, "log": []}, stream_mode="updates"):
    print("  ", update)

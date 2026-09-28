"""
Lesson 12.3: The ReAct agent built from scratch as a LangGraph graph.
This is (a simplified version of) what create_agent() builds for you.
"""

from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from course_utils import banner, get_model


@tool
def get_price(product: str) -> str:
    """Get the price in INR of a Sweet Crumbs product."""
    # Always include units! Returning a bare "90" made llama3.1 answer "$1.90".
    return {"cardamom bun": "60 INR", "masala croissant": "90 INR"}.get(product.lower(), "unknown product")


tools = [get_price]
tools_by_name = {t.name: t for t in tools}
model = get_model().bind_tools(tools)
SYSTEM = SystemMessage("You are a bakery assistant. Use tools for prices. Be brief.")


# MessagesState = {"messages": Annotated[list, add_messages]}: new messages are APPENDED
def call_model(state: MessagesState) -> dict:
    return {"messages": [model.invoke([SYSTEM] + state["messages"])]}


def call_tools(state: MessagesState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        output = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": results}


def route_after_model(state: MessagesState) -> str:
    """The heart of an agent: tool calls? -> run tools; otherwise -> done."""
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_node("tools", call_tools)
builder.add_edge(START, "model")
builder.add_conditional_edges("model", route_after_model, ["tools", END])
builder.add_edge("tools", "model")  # after tools, always go back to the model
agent = builder.compile(checkpointer=InMemorySaver())

banner("The graph (same shape as create_agent's)")
print(agent.get_graph().draw_mermaid())

banner("Run it, streaming node updates")
config = {"configurable": {"thread_id": "t1"}}
for update in agent.stream({"messages": [("user", "How much is a masala croissant?")]}, config, stream_mode="updates"):
    for node, data in update.items():
        m = data["messages"][-1]
        print(f"  [{node:5}] {m.type:4} {m.content or m.tool_calls}")

banner("Memory works too (checkpointer)")
out = agent.invoke({"messages": [("user", "And what did I just ask about?")]}, config)
print("  ", out["messages"][-1].content)

banner("Time travel: every step was checkpointed")
for i, snapshot in enumerate(agent.get_state_history(config)):
    print(f"  checkpoint {i}: {len(snapshot.values['messages'])} messages, next={snapshot.next}")

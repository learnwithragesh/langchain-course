"""
Lesson 10.2: Streaming an agent: step updates and live tokens.
"""

from langchain.agents import create_agent
from langchain.tools import tool

from course_utils import banner, get_model

HOURS = "Open 6:30 AM - 9:00 PM daily; Tuesdays close at 5:00 PM."


@tool
def store_hours() -> str:
    """Get Sweet Crumbs store opening hours."""
    return HOURS


@tool
def delivery_info(distance_km: float) -> str:
    """Check whether Sweet Crumbs can deliver to an address this many km from the store, and the fee."""
    if distance_km > 8:
        return "Sorry, we only deliver within 8 km."
    return "Delivery available. Fee Rs 49 (free for orders above Rs 999)."


agent = create_agent(get_model(), tools=[store_hours, delivery_info],
                     system_prompt="You are a helpful bakery assistant. Use tools for facts.")
question = {"messages": [{"role": "user", "content": "I'm 5 km away. Can you deliver, and until what time are you open?"}]}

banner('stream_mode="updates": one event per graph step')
for update in agent.stream(question, stream_mode="updates"):
    for node_name, data in update.items():  # node_name is 'model' or 'tools'
        last = data["messages"][-1]
        if getattr(last, "tool_calls", None):
            summary = f"wants tools {[tc['name'] for tc in last.tool_calls]}"
        else:
            summary = repr(last.content[:100])
        print(f"  [{node_name:5}] {type(last).__name__:11} {summary}")

banner('stream_mode="messages": token-by-token (for chat UIs)')
for token, metadata in agent.stream(question, stream_mode="messages"):
    # Only print text tokens produced by the model node (skip tool outputs)
    if metadata["langgraph_node"] == "model" and token.content:
        print(token.content, end="", flush=True)
print()

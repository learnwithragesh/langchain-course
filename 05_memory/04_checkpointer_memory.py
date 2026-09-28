"""
Lesson 05.4: The MODERN way to add memory, with a LangGraph checkpointer.

create_agent() is covered fully in lesson 10. Here we use it with NO tools,
so it's just a chatbot whose conversation state is saved per thread_id.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from course_utils import banner, get_model

agent = create_agent(
    model=get_model(),
    tools=[],
    system_prompt="You are a helpful assistant. Answer in 1 sentence.",
    checkpointer=InMemorySaver(),  # saves state after every step
)


def say(thread: str, text: str):
    config = {"configurable": {"thread_id": thread}}
    # We only send the NEW message. The checkpointer restores the rest.
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
    print(f"[{thread}] You: {text}")
    print(f"[{thread}] AI:  {result['messages'][-1].content}\n")


banner("Thread memory")
say("thread-1", "Remember this code word: MANGO-42.")
say("thread-1", "What was the code word?")
say("thread-2", "What was the code word?")  # new thread, so it has no memory

banner("Inspect saved state for thread-1")
state = agent.get_state({"configurable": {"thread_id": "thread-1"}})
for m in state.values["messages"]:
    print(f"  [{m.type:5}] {m.content}")

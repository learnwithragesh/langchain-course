"""
Lesson 12.4: Long-term memory with a Store, which persists ACROSS conversations (threads).

Checkpointer = remembers THIS conversation.
Store        = remembers facts about the USER forever, in every conversation.
"""

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from course_utils import banner, get_model


@dataclass
class Context:
    user_id: str


@tool
def save_preference(preference: str, runtime: ToolRuntime[Context]) -> str:
    """Save a lasting fact or preference about the user (diet, allergies, favourite items)."""
    namespace = ("preferences", runtime.context.user_id)  # like a folder per user
    # Use the normalised text as the key, so saving the same fact twice OVERWRITES instead of duplicating.
    # (With a counter-based key, llama3.1 saved "egg-free diet" 3 times in one conversation.)
    key = preference.lower().strip()
    runtime.store.put(namespace, key, {"text": preference})
    return f"Saved: {preference}"


@tool  # Pattern A only; NOT given to the agent below (see Pattern B)
def get_preferences(runtime: ToolRuntime[Context]) -> str:
    """Get everything we know about the user's preferences. Call this before recommending products."""
    items = runtime.store.search(("preferences", runtime.context.user_id))
    return "\n".join(i.value["text"] for i in items) or "No saved preferences."


BASE_PROMPT = ("You are the Sweet Crumbs assistant. When the user tells you a lasting preference "
               "(diet, allergy, favourite), call save_preference. "
               "Menu: Cardamom Bun (eggless), Masala Croissant (has egg), "
               "Millet Bread (eggless, gluten-free), Mango Cheesecake (has egg). Be brief.")


# Pattern A: rely on the model to CALL get_preferences (the tool above).
# With llama3.1 this often fails: in a new thread it asks "any dietary restrictions?" instead of looking.
#
# Pattern B (more reliable): middleware READS the store and injects memories into the system prompt
# on every model call. The model can't forget to look, because the memories are always in front of it.
@dynamic_prompt
def inject_memories(request: ModelRequest) -> str:
    items = request.runtime.store.search(("preferences", request.runtime.context.user_id))
    memories = "\n".join(f"- {i.value['text']}" for i in items) or "- (nothing yet)"
    return f"{BASE_PROMPT}\n\nKnown facts about this user (from past conversations):\n{memories}"


store = InMemoryStore()  # production: PostgresStore
agent = create_agent(
    get_model(),
    tools=[save_preference],     # recall is done by middleware, so only the write tool is needed
    context_schema=Context,
    checkpointer=InMemorySaver(),
    store=store,
    middleware=[inject_memories],
)


def chat(thread: str, text: str, user="u_ravi"):
    out = agent.invoke({"messages": [{"role": "user", "content": text}]},
                       {"configurable": {"thread_id": thread}}, context=Context(user_id=user))
    print(f"[{thread}] You: {text}\n[{thread}] AI:  {out['messages'][-1].content}\n")


banner("Conversation 1: user shares preferences")
chat("monday-chat", "Just so you know, I'm vegetarian and I don't eat eggs.")

banner("Conversation 2 (NEW thread, so no chat history), but the store remembers")
chat("friday-chat", "Recommend me something from your menu.")

banner("What's in the store")
for item in store.search(("preferences", "u_ravi")):
    print(f"  {item.namespace} {item.key}: {item.value}")

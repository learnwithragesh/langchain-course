"""
Lesson 10.5: Agentic RAG. The retriever becomes a TOOL, and the agent decides when and what to search.

Compare with lesson 08, where the chain ALWAYS retrieved, exactly once, using the raw question.
"""

from langchain.agents import create_agent
from langchain.tools import tool

from course_utils import banner, get_bakery_store, get_model

store = get_bakery_store()


@tool
def search_bakery_docs(query: str) -> str:
    """Search Sweet Crumbs' internal documents (employee handbook, customer FAQ, product list).
    Use a short, specific search query. You may call this several times with different queries."""
    docs = store.similarity_search(query, k=3)
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


agent = create_agent(
    model=get_model(),
    tools=[search_bakery_docs],
    system_prompt=("You answer questions about the Sweet Crumbs bakery. For any question about the bakery, "
                   "search the documents first and answer ONLY from what you find, citing the source file. "
                   "For general chit-chat, answer directly without searching. "
                   "If the documents don't contain the answer, say so."),
)

for q in [
    "Hi there! How are you?",                                   # should NOT search
    "What is the refund policy for custom cakes?",              # one search
    "What's the employee discount, and does it apply to catering orders?",  # maybe 1-2 searches
]:
    banner(q)
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  🔎 search: {tc['args'].get('query')!r}")
    print(f"\n  ANSWER: {result['messages'][-1].content}")

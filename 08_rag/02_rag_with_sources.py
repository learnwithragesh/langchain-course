"""
Lesson 08.2: Return the answer AND the sources, so users can verify.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from course_utils import banner, get_bakery_store, get_model

retriever = get_bakery_store().as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(f"[{i + 1}] (source: {d.metadata['source']})\n{d.page_content}" for i, d in enumerate(docs))


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the numbered context passages. After each fact, cite the passage number like [1]. "
     "If the context does not contain the answer, say you don't know.\n\nContext:\n{context}"),
    ("human", "{question}"),
])

answer_chain = prompt | get_model() | StrOutputParser()

# Pattern: retrieve ONCE, keep the docs in the dict, and build the answer from them.
rag_with_sources = (
    {"docs": retriever, "question": RunnablePassthrough()}          # {docs: [...], question: str}
    | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
    | RunnablePassthrough.assign(answer=answer_chain)              # adds "answer"
)

for q in ["Can I get a refund on a custom cake?", "What do new employees learn in training?"]:
    banner(q)
    result = rag_with_sources.invoke(q)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for i, d in enumerate(result["docs"], 1):
        label = d.metadata.get("section") or d.page_content.splitlines()[0][:50]
        print(f"  [{i}] {d.metadata['source']} :: {label}")

banner("Streaming the answer while still keeping sources")
for chunk in rag_with_sources.stream("What are the store hours?"):
    if "answer" in chunk:
        print(chunk["answer"], end="", flush=True)
print()

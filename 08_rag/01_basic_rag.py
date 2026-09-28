"""
Lesson 08.1: Basic RAG. Compare the model with and without retrieval.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from course_utils import banner, get_bakery_store, get_model

model = get_model()
retriever = get_bakery_store().as_retriever(search_kwargs={"k": 3})


def format_docs(docs) -> str:
    """Turn a list of Documents into one string for the prompt."""
    return "\n\n".join(f"<doc source='{d.metadata['source']}'>\n{d.page_content}\n</doc>" for d in docs)


rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the Sweet Crumbs bakery assistant. Answer the question using ONLY the context below. "
     "If the answer is not in the context, say \"I don't know based on the available documents.\" "
     "Be concise.\n\n<context>\n{context}\n</context>"),
    ("human", "{question}"),
])

# The whole RAG pipeline in LCEL:
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

no_rag_chain = model | StrOutputParser()

questions = [
    "How many days of paid annual leave do Sweet Crumbs employees get?",
    "What is the delivery fee at Sweet Crumbs?",
    "What is Sweet Crumbs' best-selling product?",
    "Does Sweet Crumbs sell pizza?",  # not in the documents
]

for q in questions:
    banner(q)
    print("WITHOUT RAG:", no_rag_chain.invoke(q)[:300])
    print("\nWITH RAG:   ", rag_chain.invoke(q))

banner("Debugging tip: always look at what was retrieved")
for d in retriever.invoke("What is the delivery fee?"):
    print(f"  [{d.metadata['source']}] {d.page_content[:80]!r}")

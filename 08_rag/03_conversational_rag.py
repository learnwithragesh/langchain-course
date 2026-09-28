"""
Lesson 08.3: Conversational RAG. Follow-up questions that depend on chat history.

Problem: "What about sick leave?" retrieves badly, because the retriever doesn't
know we were talking about employee leave.
Fix: first ask the LLM to CONDENSE (history + follow-up) into a standalone question.
"""

from langchain.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from course_utils import banner, get_bakery_store, get_model

model = get_model()
retriever = get_bakery_store().as_retriever(search_kwargs={"k": 3})

# Step 1: rewrite the follow-up into a standalone question
condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and a follow-up question, rewrite the follow-up as a standalone "
               "question that makes sense without the history. Output ONLY the rewritten question."),
    MessagesPlaceholder("history"),
    ("human", "Follow-up question: {question}"),
])
condense = condense_prompt | model | StrOutputParser()


def standalone_question(x: dict) -> str:
    if not x["history"]:  # first turn: nothing to condense
        return x["question"]
    q = condense.invoke(x)
    print(f"   [condensed] {x['question']!r} -> {q!r}")
    return q


# Step 2: normal RAG on the standalone question, but answer with history for a natural reply
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Sweet Crumbs assistant. Answer ONLY from the context. "
               "If unsure, say you don't know. Be concise.\n\n<context>\n{context}\n</context>"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

conversational_rag = (
    RunnablePassthrough.assign(standalone=RunnableLambda(standalone_question))
    | RunnablePassthrough.assign(
        context=lambda x: "\n\n".join(d.page_content for d in retriever.invoke(x["standalone"])))
    | answer_prompt
    | model
    | StrOutputParser()
)

history = []
for q in [
    "How many days of paid annual leave do employees get?",
    "And what about sick leave?",
    "How much notice do I need to give for the first one?",
]:
    banner(q)
    answer = conversational_rag.invoke({"question": q, "history": history})
    print("AI:", answer)
    history += [HumanMessage(q), AIMessage(answer)]

banner("Why condensing matters: raw follow-up vs. condensed retrieval")
vague = "How much notice do I need to give for the first one?"
print(f"raw {vague!r} retrieves:")
for d in retriever.invoke(vague):
    print("   ", d.page_content.splitlines()[0][:60])
print("condensed 'notice for taking annual paid leave' retrieves:")
for d in retriever.invoke("What is the required notice for taking annual paid leave?"):
    print("   ", d.page_content.splitlines()[0][:60])

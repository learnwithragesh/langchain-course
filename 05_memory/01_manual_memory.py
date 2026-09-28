"""
Lesson 05.1: Memory with no magic. A list of messages plus a MessagesPlaceholder.
"""

from langchain.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from course_utils import banner, get_model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers to 1-2 sentences."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | get_model() | StrOutputParser()

history: list = []  # <- this list IS the memory


def chat(user_text: str) -> str:
    answer = chain.invoke({"history": history, "input": user_text})
    # Save both sides of the exchange for next time
    history.append(HumanMessage(user_text))
    history.append(AIMessage(answer))
    return answer


banner("A scripted conversation")
for msg in [
    "Hi, I'm learning LangChain. My favourite language is Python.",
    "I'm building a chatbot for my bakery called 'Sweet Crumbs'.",
    "What's my bakery called, and what language should I build it in?",
]:
    print(f"You: {msg}")
    print(f"AI:  {chat(msg)}\n")

banner(f"The memory now holds {len(history)} messages")
for m in history:
    print(f"  [{m.type:5}] {m.content[:70]}")

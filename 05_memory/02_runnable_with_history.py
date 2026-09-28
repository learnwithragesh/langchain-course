"""
Lesson 05.2: RunnableWithMessageHistory. Automatic history, separate for each session.

!! DEPRECATED since langchain-core 1.6.4. You will see LangChainDeprecationWarning when you run this.
   It is taught because lots of existing code and tutorials use it.
   For NEW code use the checkpointer approach in 04_checkpointer_memory.py.
"""

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from course_utils import banner, get_model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep answers to 1 sentence."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
chain = prompt | get_model() | StrOutputParser()

# session_id -> history object. In production this would be Redis, Postgres, etc.
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chat = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",      # which input key is the new user message
    history_messages_key="history",  # which placeholder receives the history
)


def say(session: str, text: str):
    reply = chat.invoke({"input": text}, config={"configurable": {"session_id": session}})
    print(f"[{session}] You: {text}\n[{session}] AI:  {reply}\n")


banner("Two users, two separate memories")
say("alice", "My favourite colour is green.")
say("bob", "My favourite colour is orange.")
say("alice", "What is my favourite colour?")
say("bob", "What is my favourite colour?")

banner("What's inside the store")
for sid, hist in store.items():
    print(f"{sid}: {len(hist.messages)} messages")

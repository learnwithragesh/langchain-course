"""
Lesson 05: an INTERACTIVE chatbot with memory. Chat with it in your terminal.

    python 05_memory/chatbot.py

Commands:  /history  show what's being sent   |   /quit  exit
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from course_utils import CHAT_MODEL, get_model

MAX_TOKENS = 2000  # memory budget: try 100 to watch it forget

model = get_model(temperature=0.5)
trimmer = trim_messages(max_tokens=MAX_TOKENS, strategy="last", include_system=True,
                        start_on="human", token_counter=count_tokens_approximately)
history = [SystemMessage("You are a friendly, concise assistant.")]

print(f"Chatbot using {CHAT_MODEL} (memory budget {MAX_TOKENS} tokens). Type /quit to exit.\n")
while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user:
        continue
    if user == "/quit":
        break
    if user == "/history":
        for m in trimmer.invoke(history):
            print(f"   [{m.type}] {m.content[:80]}")
        continue

    history.append(HumanMessage(user))
    print("AI:  ", end="")
    reply = ""
    for chunk in model.stream(trimmer.invoke(history)):  # stream for a nice typing effect
        print(chunk.content, end="", flush=True)
        reply += chunk.content
    print("\n")
    history.append(AIMessage(reply))

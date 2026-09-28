"""
Lesson 01.2: Message types, system prompts, and manual conversation history.
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from course_utils import banner, get_model

model = get_model()

banner("A system message sets behaviour")
messages = [
    SystemMessage("You are a concise tutor. Always answer in exactly 2 sentences."),
    HumanMessage("What is photosynthesis?"),
]
print(model.invoke(messages).content)

banner("Same thing with the tuple shorthand")
print(model.invoke([
    ("system", "You explain things to a 5-year-old."),
    ("user", "What is photosynthesis?"),
]).content)

banner("Multi-turn: YOU carry the history, because the model is stateless")
history = [
    SystemMessage("You are a friendly assistant."),
    HumanMessage("Hi! My name is Ragesh and I love cricket."),
]
first = model.invoke(history)
print("AI:", first.content)

history.append(first)  # add the AI's reply to history
history.append(HumanMessage("What is my name and what sport do I like?"))
second = model.invoke(history)
print("AI:", second.content)

banner("Proof of statelessness: ask the same question WITHOUT history")
print("AI:", model.invoke("What is my name and what sport do I like?").content)

banner("You can even fake an AI turn (useful for few-shot examples)")
print(model.invoke([
    HumanMessage("Translate to French: cat"),
    AIMessage("chat"),
    HumanMessage("Translate to French: dog"),
]).content)

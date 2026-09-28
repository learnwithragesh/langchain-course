"""
Lesson 05.3: Keeping history within a budget using trim_messages.
"""

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

from course_utils import banner, get_model

history = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("Hi, my name is Ragesh."),
    AIMessage("Hello Ragesh! How can I help?"),
    HumanMessage("I live in Coimbatore."),
    AIMessage("Coimbatore is a lovely city, known for textiles."),
    HumanMessage("I work as a data engineer."),
    AIMessage("Great! Data engineering is in high demand."),
    HumanMessage("I'm learning LangChain this month."),
    AIMessage("Excellent choice for building LLM apps."),
]

print("Total approx tokens:", count_tokens_approximately(history))

banner("Keep only the LAST ~60 tokens (plus the system message)")
trimmer = trim_messages(
    max_tokens=60,
    strategy="last",          # keep the most recent messages
    token_counter=count_tokens_approximately,  # or pass a model to count exactly
    include_system=True,      # never drop the system prompt
    start_on="human",         # the kept history must start with a human turn
)
trimmed = trimmer.invoke(history)
for m in trimmed:
    print(f"  [{m.type:6}] {m.content}")

banner("Effect: recent facts remembered, old facts forgotten")
model = get_model()
question = HumanMessage("What is my name and what am I learning? If you don't know, say so.")
print("FULL history   ->", model.invoke(history + [question]).content)
print("TRIMMED history->", model.invoke(trimmed + [question]).content)

banner("Trimmer is a Runnable, so put it straight into a chain")
chain = trimmer | model
print(chain.invoke(history + [HumanMessage("Where do I live?")]).content)

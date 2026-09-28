"""
Lesson 02.4: MessagesPlaceholder inserts a whole LIST of messages into a template.
This is the building block for chat memory (lesson 05).
"""

from langchain.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from course_utils import banner, get_model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the conversation so far."),
    MessagesPlaceholder("history"),  # a list of messages goes here
    ("human", "{question}"),
])

history = [
    HumanMessage("I'm planning a trip to Kerala in December."),
    AIMessage("Lovely! December is a great time: pleasant weather, and it's festival season."),
]

banner("Rendered prompt")
for m in prompt.invoke({"history": history, "question": "What should I pack?"}).to_messages():
    print(f"  [{m.type:6}] {m.content}")

banner("Model answer (it knows about Kerala + December from history)")
chain = prompt | get_model() | StrOutputParser()
print(chain.invoke({"history": history, "question": "What should I pack? 5 items only."}))

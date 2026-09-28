"""
Lesson 04.3: Routing. Classify the input, then send it to a specialised chain.
Structured output (L03) and LCEL (L04) working together.
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from course_utils import banner, get_model

model = get_model()
parser = StrOutputParser()


# 1. A classifier that returns a label
class Route(BaseModel):
    topic: Literal["math", "coding", "general"]


classifier = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the user's question as math, coding or general."),
        ("human", "{question}"),
    ])
    | model.with_structured_output(Route, method="json_schema")
)

# 2. Specialist chains
math_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a maths tutor. Show the working step by step, briefly."),
    ("human", "{question}"),
]) | model | parser

code_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a senior Python developer. Answer with a short code example."),
    ("human", "{question}"),
]) | model | parser

general_chain = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Answer in 2 sentences."),
    ("human", "{question}"),
]) | model | parser


# 3. The router: a function that RETURNS a Runnable. LCEL then runs that Runnable.
def route(info: dict):
    topic = info["route"].topic
    print(f"   [router] -> {topic}")
    return {"math": math_chain, "coding": code_chain}.get(topic, general_chain)


full_chain = RunnablePassthrough.assign(route=classifier) | RunnableLambda(route)

for q in [
    "What is 15% of 240?",
    "How do I reverse a list in Python?",
    "Who painted the Mona Lisa?",
]:
    banner(q)
    print(full_chain.invoke({"question": q}))

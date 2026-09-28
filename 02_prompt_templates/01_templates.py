"""
Lesson 02.1: Prompt templates on their own, with no model yet.
Seeing exactly what a template produces removes all the "magic".
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from course_utils import banner

banner("PromptTemplate -> a single string")
pt = PromptTemplate.from_template("Tell me a {adjective} fact about {topic}.")
print("input_variables:", pt.input_variables)
value = pt.invoke({"adjective": "surprising", "topic": "octopuses"})
print("type:", type(value).__name__)
print("as string:", value.to_string())

banner("ChatPromptTemplate -> a list of messages")
chat_pt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Keep answers under {n_words} words."),
    ("human", "{question}"),
])
print("input_variables:", chat_pt.input_variables)
value = chat_pt.invoke({"role": "patient maths teacher", "n_words": 50, "question": "What is a prime number?"})
for m in value.to_messages():
    print(f"  [{m.type:6}] {m.content}")

banner("partial(): pre-fill some variables")
teacher = chat_pt.partial(role="patient maths teacher", n_words=30)
print("remaining input_variables:", teacher.input_variables)
for m in teacher.invoke({"question": "What is pi?"}).to_messages():
    print(f"  [{m.type:6}] {m.content}")

banner("Literal braces must be doubled: {{ }}")
json_pt = ChatPromptTemplate.from_messages([
    ("system", 'Reply ONLY with JSON like {{"answer": "..."}}'),
    ("human", "{q}"),
])
print(json_pt.invoke({"q": "hi"}).to_messages()[0].content)

banner("Missing variable -> error (a good thing!)")
try:
    chat_pt.invoke({"role": "chef"})
except KeyError as e:
    print("KeyError:", e)

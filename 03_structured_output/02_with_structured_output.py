"""
Lesson 03.2: with_structured_output(), the modern and recommended approach.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from course_utils import banner, get_model

model = get_model()


class Person(BaseModel):
    """A person mentioned in the text."""

    name: str = Field(description="The person's full name")
    age: Optional[int] = Field(default=None, description="Age in years, if stated")
    city: Optional[str] = Field(default=None, description="City they live in, if stated")
    email: Optional[str] = Field(default=None, description="Email address, if stated")
    hobbies: list[str] = Field(default_factory=list, description="Hobbies or interests")


class People(BaseModel):
    """All people mentioned in the text."""

    people: list[Person]


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise information-extraction system. Extract only facts stated in the text. "
               "Use null for anything not mentioned."),
    ("human", "{text}"),
])

banner("Single object")
extract_person = prompt | model.with_structured_output(Person, method="json_schema")
p = extract_person.invoke({"text": "Hi, I'm Ravi Kumar, 29, from Chennai. I love chess and trekking."})
print(type(p).__name__, "->", p)
print("Real Python object:  p.age + 1 =", p.age + 1 if p.age else None)

banner("Missing info -> None (thanks to Optional)")
print(extract_person.invoke({"text": "Meena works as a designer."}))

banner("A list of objects in one call")
extract_people = prompt | model.with_structured_output(People, method="json_schema")
text = ("At the meetup, Arjun (34, Bengaluru) talked about Rust. "
        "Priya from Pune, reachable at priya@example.com, loves painting. "
        "Old John, 70, just listened.")
for person in extract_people.invoke({"text": text}).people:
    print("  ", person.model_dump())

banner("include_raw=True for debugging")
debug = prompt | model.with_structured_output(Person, method="json_schema", include_raw=True)
out = debug.invoke({"text": "Kavya is 22."})
print("keys:", list(out.keys()))
print("raw text from model:", out["raw"].content)
print("parsed:", out["parsed"])
print("parsing_error:", out["parsing_error"])

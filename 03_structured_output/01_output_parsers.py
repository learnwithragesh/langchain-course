"""
Lesson 03.1: Output parsers, the prompt-based (older, portable) approach.
"""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from course_utils import banner, get_model

model = get_model()

banner("StrOutputParser: AIMessage -> str")
chain = model | StrOutputParser()
out = chain.invoke("Say 'hello' in Tamil. Just the word.")
print(type(out).__name__, "->", out, "| is a str?", isinstance(out, str))  # TextAccessor is a str subclass


class Movie(BaseModel):
    title: str = Field(description="movie title")
    year: int = Field(description="release year")
    genres: list[str] = Field(description="list of genres")


banner("JsonOutputParser with format instructions")
parser = JsonOutputParser(pydantic_object=Movie)

# The parser can generate instructions describing the JSON it expects:
print("Format instructions injected into the prompt:\n")
print(parser.get_format_instructions()[:400], "...\n")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's request.\n{format_instructions}"),
    ("human", "{request}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | model | parser
result = chain.invoke({"request": "Give me details of the movie Inception."})
print(type(result).__name__, "->", result)
print("Access like a dict:", result.get("title"), result.get("year"))

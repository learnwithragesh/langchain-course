"""
Lesson 04.1: The pipe operator and what really happens inside.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from course_utils import banner, get_model

model = get_model()
prompt = ChatPromptTemplate.from_template("Give one fun fact about {animal}. One sentence.")
parser = StrOutputParser()

banner("The chain vs. doing it by hand: identical results")
chain = prompt | model | parser
print("chain type:", type(chain).__name__)
print("chain:  ", chain.invoke({"animal": "elephants"}))

# The same thing, manually, step by step:
step1 = prompt.invoke({"animal": "elephants"})
step2 = model.invoke(step1)
step3 = parser.invoke(step2)
print("manual: ", step3)

banner("Plain functions become steps (RunnableLambda)")


def shout(text: str) -> str:
    return text.upper()


def count_words(text: str) -> dict:
    return {"text": text, "words": len(text.split())}


chain2 = prompt | model | parser | shout | count_words  # functions are auto-wrapped
print(chain2.invoke({"animal": "owls"}))

banner("Input normalisation: accept a bare string instead of a dict")
# RunnableLambda lets the chain start with a function (the first item must be a Runnable)
chain3 = RunnableLambda(lambda s: {"animal": s}) | prompt | model | parser
print(chain3.invoke("penguins"))

banner("Chaining two LLM calls: the output of one is the input of the next")
write = ChatPromptTemplate.from_template("Write a 2-sentence product description for: {product}") | model | parser
translate = ChatPromptTemplate.from_template("Translate to Hindi:\n{text}") | model | parser

pipeline = write | (lambda desc: {"text": desc}) | translate
print(pipeline.invoke({"product": "a solar-powered water bottle"}))

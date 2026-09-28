"""
Lesson 02.2: Your first CHAIN: prompt | model | parser.

The `|` operator connects Runnables: the output of the left side becomes the
input of the right side. Lesson 04 explains this in depth. For now, just see it work.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from course_utils import banner, get_model

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} teacher. Explain in 3 bullet points for a {level}."),
    ("human", "Explain: {concept}"),
])
model = get_model()
parser = StrOutputParser()  # AIMessage -> plain str

chain = prompt | model | parser
#        dict  -> messages -> AIMessage -> str

banner("Same chain, different inputs")
print(chain.invoke({"domain": "physics", "level": "10 year old", "concept": "gravity"}))
print("-" * 40)
print(chain.invoke({"domain": "computer science", "level": "senior engineer", "concept": "hash tables"}))

banner("Chains are Runnables, so they stream and batch too")
for piece in chain.stream({"domain": "biology", "level": "beginner", "concept": "DNA"}):
    print(piece, end="", flush=True)
print()

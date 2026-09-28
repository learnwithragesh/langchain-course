"""
Lesson 04.2: Running things in parallel and carrying data through a chain.
"""

import time
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from course_utils import banner, get_model

model = get_model()
parser = StrOutputParser()


def ask(template: str):
    return ChatPromptTemplate.from_template(template) | model | parser


banner("RunnableParallel: three LLM calls at once, results in a dict")
analysis = RunnableParallel(
    pros=ask("List 3 pros of {tech}. Short bullets only."),
    cons=ask("List 3 cons of {tech}. Short bullets only."),
    one_liner=ask("Describe {tech} in 10 words."),
)
start = time.time()
result = analysis.invoke({"tech": "electric cars"})
print(f"(took {time.time() - start:.1f}s for 3 calls)\n")
for key, value in result.items():
    print(f"--- {key} ---\n{value}\n")

banner("A dict inside a chain is automatically a RunnableParallel")
summary = {
    "pros": ask("List 3 pros of {tech}. Short bullets only."),
    "cons": ask("List 3 cons of {tech}. Short bullets only."),
} | ask("Given these pros:\n{pros}\nand cons:\n{cons}\nGive a one-sentence verdict.")
print(summary.invoke({"tech": "remote work"}))

banner("RunnablePassthrough: forward the input unchanged")
chain = RunnableParallel(original=RunnablePassthrough(), upper=lambda x: x["word"].upper())
print(chain.invoke({"word": "langchain"}))

banner("RunnablePassthrough.assign: KEEP the input and ADD keys")
# Very common pattern: enrich a dict step by step
enrich = (
    RunnablePassthrough.assign(length=lambda x: len(x["text"]))
    | RunnablePassthrough.assign(summary=ask("Summarise in 5 words: {text}"))
)
print(enrich.invoke({"text": "LangChain is a framework for developing applications powered by large language models."}))

banner("itemgetter: route specific keys to specific places")
qa = (
    {"question": itemgetter("question"), "style": itemgetter("style")}
    | ask("Answer the question in the style of {style}: {question}")
)
print(qa.invoke({"question": "Why is the sky blue?", "style": "a Shakespearean poem (4 lines)", "ignored": 123}))

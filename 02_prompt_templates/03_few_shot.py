"""
Lesson 02.3: Few-shot prompting. Teach a format by showing examples.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from course_utils import banner, get_model

# The examples we want the model to imitate
examples = [
    {"word": "happy", "output": "happy -> sad | emoji: 😀 | syllables: 2"},
    {"word": "tall", "output": "tall -> short | emoji: 🦒 | syllables: 1"},
]

# How ONE example is rendered: a human turn followed by an AI turn
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{word}"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You give the antonym, an emoji and the syllable count, in the exact format shown."),
    few_shot,  # the examples are inserted here as fake conversation turns
    ("human", "{word}"),
])

banner("What the model actually receives")
for m in final_prompt.invoke({"word": "fast"}).to_messages():
    print(f"  [{m.type:6}] {m.content}")

chain = final_prompt | get_model() | StrOutputParser()

banner("Results: the format is copied from the examples")
for w in ["fast", "beautiful", "hot"]:
    print("  ", chain.invoke({"word": w}))

banner("Compare: zero-shot (no examples)")
zero = ChatPromptTemplate.from_messages([
    ("system", "You give the antonym, an emoji and the syllable count. Example Format - Print Exactly Like: happy -> sad | emoji: 😀 | syllables: 2"),
    ("human", "{word}"),
]) | get_model() | StrOutputParser()
print("  ", zero.invoke({"word": "fast"}))

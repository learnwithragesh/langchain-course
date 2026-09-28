"""
Lesson 04.4: Production helpers: retries, fallbacks, config, and inspecting a chain.
"""

import random

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from course_utils import banner, get_model

model = get_model()
prompt = ChatPromptTemplate.from_template("Say '{word}' backwards. Only the result.")

banner("with_retry(): survive flaky steps")
attempts = {"n": 0}


def flaky(x):
    attempts["n"] += 1
    if random.random() < 0.6:
        print(f"   attempt {attempts['n']}: simulated network error")
        raise ConnectionError("boom")
    print(f"   attempt {attempts['n']}: ok")
    return x


random.seed(3)
robust = RunnableLambda(flaky).with_retry(stop_after_attempt=5, wait_exponential_jitter=False)
print("result:", robust.invoke("hello"))

banner("with_fallbacks(): use a backup model if the primary fails")
broken = init_chat_model("ollama:model-that-does-not-exist")
safe_model = broken.with_fallbacks([model])
chain = prompt | safe_model | StrOutputParser()
print("result:", chain.invoke({"word": "stressed"}))

banner("Inspect a chain: input schema and graph")
chain = prompt | model | StrOutputParser()
print("input schema:", chain.get_input_jsonschema())
print("\nMermaid diagram (paste into https://mermaid.live):\n")
print(chain.get_graph().draw_mermaid())

banner("config: name runs, tag them, limit concurrency")
out = chain.with_config(run_name="reverse-word").batch(
    [{"word": w} for w in ["level", "drawer", "stop"]],
    config={"max_concurrency": 2, "tags": ["lesson04"]},
)
print(out)

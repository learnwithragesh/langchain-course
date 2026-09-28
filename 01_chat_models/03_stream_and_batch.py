"""
Lesson 01.3: The three ways to call ANY Runnable: invoke, stream, and batch (plus async).
"""

import asyncio
import time

from course_utils import banner, get_model

model = get_model()

banner("stream(): print tokens as they arrive")
for chunk in model.stream("Write a 4-line poem about the monsoon."):
    # Each chunk is an AIMessageChunk with a small piece of text
    print(chunk.content, end="", flush=True)
print()

banner("Chunks can be added together with +")
full = None
for chunk in model.stream("Name three primary colours, comma separated."):
    full = chunk if full is None else full + chunk
print("combined:", full.content)

banner("batch(): several inputs at once")
questions = [
    "Capital of France? One word.",
    "Capital of India? One word.",
    "Capital of Brazil? One word.",
]
start = time.time()
answers = model.batch(questions)  # runs in parallel threads
for q, a in zip(questions, answers):
    print(f"  {q:32} -> {a.content}")
print(f"  took {time.time() - start:.1f}s")

banner("Async: ainvoke(), used inside web servers (FastAPI etc.)")


async def main():
    # gather runs the calls concurrently
    results = await asyncio.gather(
        model.ainvoke("2+2? Just the number."),
        model.ainvoke("3*3? Just the number."),
    )
    print("  ", [r.content for r in results])


asyncio.run(main())

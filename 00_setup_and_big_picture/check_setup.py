"""
Lesson 00: Verify the environment works end to end.

Run:  python 00_setup_and_big_picture/check_setup.py
"""

import importlib.metadata as md

from course_utils import CHAT_MODEL, EMBED_MODEL, banner, get_embeddings, get_model

banner("1. Installed package versions")
for pkg in ["langchain", "langchain-core", "langgraph", "langchain-ollama", "langchain-anthropic"]:
    print(f"  {pkg:22} {md.version(pkg)}")

banner(f"2. Chat model: {CHAT_MODEL}")
model = get_model()
reply = model.invoke("In one short sentence, what is LangChain?")
print("  ", reply.content)

banner(f"3. Embedding model: {EMBED_MODEL}")
vector = get_embeddings().embed_query("hello world")
print(f"   'hello world' -> a vector of {len(vector)} numbers, first 5: {[round(x, 3) for x in vector[:5]]}")

print("\nAll good! Continue with lesson 01.")

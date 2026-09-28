"""
Lesson 01.4: How model parameters change behaviour.
"""

from course_utils import CHAT_MODEL, banner, get_model

prompt = "Invent a name for a new coffee shop. Reply with the name only."

banner("temperature=0: (nearly) the same answer every time")
cold = get_model(temperature=0)
for _ in range(3):
    print("  ", cold.invoke(prompt).content)

banner("temperature=1.0: more random and creative")
hot = get_model(temperature=1.0)
for _ in range(3):
    print("  ", hot.invoke(prompt).content)

banner("Limiting output length")
# Cloud providers call this max_tokens. Ollama calls it num_predict.
# The reply gets cut off mid-sentence, so a limit is a hard stop, not a "be brief" instruction.
short = get_model(num_predict=15) if CHAT_MODEL.startswith("ollama") else get_model(max_tokens=15)
reply = short.invoke("Explain the history of the internet.")
print("  ", reply.content)
print("   done_reason / stop_reason:",
      reply.response_metadata.get("done_reason") or reply.response_metadata.get("stop_reason"))

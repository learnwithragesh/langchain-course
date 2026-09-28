"""
Lesson 01.1: Your first model call, and what comes back.
"""

from course_utils import banner, get_model

model = get_model()

banner("invoke() with a plain string")
response = model.invoke("What is the capital of Japan? Answer in one word.")

# The result is NOT a string. It's an AIMessage object.
print("type:              ", type(response).__name__)
print("content:           ", response.content)
print("usage_metadata:    ", response.usage_metadata)  # token counts
print("response_metadata keys:", list(response.response_metadata.keys()))

banner("The full object (notice id, metadata, tool_calls=[])")
print(repr(response))

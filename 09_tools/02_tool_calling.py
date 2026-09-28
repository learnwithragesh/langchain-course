"""
Lesson 09.2: bind_tools(). The model DECIDES which tool to call, but doesn't run it.
"""

from langchain.tools import tool

from course_utils import banner, get_model

PRICES = {"filter coffee tres leches": 650, "jaggery banana bread": 220, "cardamom bun": 60,
          "masala croissant": 90, "mango cheesecake": 720}


@tool
def get_price(product: str) -> str:
    """Look up the price in INR of a Sweet Crumbs bakery product by its name."""
    return str(PRICES.get(product.lower().strip(), "unknown product"))


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers exactly."""
    return a * b


model = get_model()
model_with_tools = model.bind_tools([get_price, multiply])

for question in [
    "How much does a cardamom bun cost?",
    "What is 1234 * 5678?",
    "What is the capital of France?",  # no tool needed
    "What do a masala croissant and a mango cheesecake cost?",  # parallel calls?
]:
    banner(question)
    msg = model_with_tools.invoke(question)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  MODEL WANTS: {tc['name']}({tc['args']})   id={tc['id'][:8]}")
        print("  content:", repr(msg.content))  # usually empty when calling tools
    else:
        print("  NO TOOL. Direct answer:", msg.content)

banner("Running a requested tool call gives a ready-made ToolMessage")
msg = model_with_tools.invoke("Price of jaggery banana bread?")
tool_msg = get_price.invoke(msg.tool_calls[0])  # pass the whole tool_call dict
print(repr(tool_msg))

banner("tool_choice: force the model to use a tool")
forced = model.bind_tools([get_price], tool_choice="get_price")
print(forced.invoke("I'd like something sweet").tool_calls)

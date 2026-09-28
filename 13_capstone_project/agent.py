"""
Capstone: assembling the agent. Every line maps to a lesson.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from tools import ALL_TOOLS, Context

from course_utils import get_model

BASE_PROMPT = """You are "Crumbs", the friendly assistant of Sweet Crumbs bakery (Tamil Nadu & Kerala).

Rules:
- For ANY question about the bakery (products, prices, delivery, refunds, policies), call search_knowledge_base
  first and answer ONLY from the results. If the answer isn't there, say you don't know.
- For prices of an order, call quote_order. Never do arithmetic yourself.
- Only call place_order after the customer clearly says they want to order.
- When the customer mentions a lasting preference (diet, allergy, favourite), call save_preference.
- Be warm and concise (2-4 sentences). Use INR for money."""


@dynamic_prompt  # L11 + L12: inject who the user is and their long-term memories
def personalised_prompt(request: ModelRequest) -> str:
    ctx: Context = request.runtime.context
    prefs = request.runtime.store.search(("preferences", ctx.user_id))
    memory = "\n".join(f"- {p.value['text']}" for p in prefs) or "- (none yet)"
    return f"{BASE_PROMPT}\n\nCustomer name: {ctx.user_name}\nKnown preferences:\n{memory}"


@wrap_tool_call  # L11: a crashing tool becomes a message, not an exception
def tool_errors_to_messages(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(f"Tool error: {e}", tool_call_id=request.tool_call["id"])


def build_agent(checkpointer, store):
    model = get_model()
    return create_agent(
        model=model,                                    # L01
        tools=ALL_TOOLS,                                # L09
        context_schema=Context,                         # L10
        checkpointer=checkpointer,                      # L05/L12: conversation memory
        store=store,                                    # L12: long-term memory
        middleware=[
            personalised_prompt,
            tool_errors_to_messages,
            SummarizationMiddleware(model=model, trigger=("messages", 30), keep=("messages", 10)),
            ToolCallLimitMiddleware(run_limit=6),       # no runaway loops
            HumanInTheLoopMiddleware(interrupt_on={     # money moves need a human
                "place_order": {"allowed_decisions": ["approve", "reject"]},
            }),
        ],
    )

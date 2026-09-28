"""
Lesson 11.3: Human-in-the-loop. Pause before sensitive actions and let a human decide.

Set INTERACTIVE = True to make the decisions yourself in the terminal.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from course_utils import banner, get_model

INTERACTIVE = False


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's details by ID."""
    return f"Order {order_id}: Mango Cheesecake, 720 INR, delivered 2026-09-25, customer reported it was damaged."


@tool
def issue_refund(order_id: str, amount_inr: float, reason: str) -> str:
    """Issue a refund to the customer. This moves real money!"""
    return f"✅ Refunded {amount_inr} INR for order {order_id} ({reason})."


agent = create_agent(
    get_model(),
    tools=[lookup_order, issue_refund],
    system_prompt="You are a support agent. Look up the order first, then refund if appropriate. Be brief.",
    checkpointer=InMemorySaver(),  # REQUIRED: pausing = saving state
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},  # needs approval
        "lookup_order": False,                                               # read-only, runs freely
    })],
)


def run_case(thread_id: str, scripted_decision: dict):
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content":
                                         "My order A-778 arrived damaged. Please refund me."}]}, config)

    while "__interrupt__" in result:  # the agent is PAUSED, waiting for a human
        request = result["__interrupt__"][0].value
        for action in request["action_requests"]:
            print(f"⏸️  APPROVAL NEEDED: {action['name']}({action['args']})")

        if INTERACTIVE:
            choice = input("   approve / reject / edit? ").strip()
            if choice == "reject":
                decision = {"type": "reject", "message": input("   reason: ")}
            elif choice == "edit":
                args = dict(request["action_requests"][0]["args"], amount_inr=float(input("   new amount: ")))
                decision = {"type": "edit", "edited_action": {"name": "issue_refund", "args": args}}
            else:
                decision = {"type": "approve"}
        else:
            decision = scripted_decision
        print(f"👤 HUMAN DECISION: {decision}")

        # Resume from the saved checkpoint with the decision
        result = agent.invoke(Command(resume={"decisions": [decision]}), config)

    for m in result["messages"]:
        if m.type == "tool":
            print(f"🔧 TOOL [{m.name}] {m.content}")
    print(f"🤖 FINAL: {result['messages'][-1].content}")


banner("Case 1: human APPROVES")
run_case("case-1", {"type": "approve"})

banner("Case 2: human REJECTS with a reason")
run_case("case-2", {"type": "reject", "message": "Refund denied: please send a photo of the damage first."})

banner("Case 3: human EDITS the amount (partial refund)")
run_case("case-3", {"type": "edit", "edited_action": {
    "name": "issue_refund", "args": {"order_id": "A-778", "amount_inr": 360, "reason": "partial refund - damaged"}}})

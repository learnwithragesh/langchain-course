"""
Capstone: "Crumbs", the Sweet Crumbs bakery assistant (a terminal app).

    python 13_capstone_project/app.py                 # chat interactively
    python 13_capstone_project/app.py --user meena    # a different customer
    python 13_capstone_project/app.py --demo          # scripted demo (no typing needed)

Memory is saved in data/capstone_memory.sqlite, so quit, restart, and it still remembers you.
Commands inside the chat:  /new  start a new conversation thread   |   /quit
"""

import argparse
import os
import uuid

from agent import build_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command
from tools import Context

from course_utils import CHAT_MODEL, DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "capstone_memory.sqlite")


def ask_human(action: dict, auto_approve: bool | None) -> dict:
    print(f"\n   ⏸️  APPROVAL NEEDED: {action['name']}({action['args']})")
    if auto_approve is not None:
        ok = auto_approve
        print(f"   (demo) {'approved' if ok else 'rejected'}")
    else:
        ok = input("   Approve this order? [y/N] ").strip().lower() == "y"
    return {"type": "approve"} if ok else {"type": "reject", "message": "Customer did not confirm the order."}


def run_turn(agent, text: str, config: dict, ctx: Context, auto_approve=None):
    """Send one user message, show tool activity, handle approvals, print the answer."""
    payload = {"messages": [{"role": "user", "content": text}]}
    while True:
        interrupted = None
        for update in agent.stream(payload, config, context=ctx, stream_mode="updates"):
            if "__interrupt__" in update:
                interrupted = update["__interrupt__"][0].value
                continue
            for node, data in update.items():
                if not data or "messages" not in data:
                    continue
                msg = data["messages"][-1]
                if node == "model" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"   🔧 {tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc['args'].items())})")
                elif node == "model" and msg.content:
                    print(f"\n🧁 Crumbs: {msg.content}\n")
        if not interrupted:
            return
        decisions = [ask_human(a, auto_approve) for a in interrupted["action_requests"]]
        payload = Command(resume={"decisions": decisions})  # resume the paused agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="ravi")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    ctx = Context(user_id=f"u_{args.user}", user_name=args.user.title())
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer, SqliteStore.from_conn_string(DB_PATH) as store:
        store.setup()
        agent = build_agent(checkpointer, store)
        thread_id = f"{ctx.user_id}-main"
        config = {"configurable": {"thread_id": thread_id}}
        print(f"🧁 Sweet Crumbs assistant  |  model: {CHAT_MODEL}  |  customer: {ctx.user_name}  |  thread: {thread_id}")

        if args.demo:
            script = [
                ("Hi! I'm eggless, so please keep that in mind for the future.", None),
                ("Do you deliver, and how much does it cost?", None),
                ("How much would 3 cardamom buns and 1 jaggery banana bread cost?", None),
                ("Great, please place that order.", True),
                ("What's your refund policy on custom cakes?", None),
            ]
            for text, approve in script:
                print(f"👤 {ctx.user_name}: {text}")
                run_turn(agent, text, config, ctx, auto_approve=approve)
            # Prove long-term memory: a brand-new thread still knows the preference
            new_config = {"configurable": {"thread_id": f"{ctx.user_id}-{uuid.uuid4().hex[:6]}"}}
            print("--- new conversation thread (no chat history) ---")
            text = "Can you suggest a bread for me?"
            print(f"👤 {ctx.user_name}: {text}")
            run_turn(agent, text, new_config, ctx)
            return

        while True:
            try:
                text = input(f"👤 {ctx.user_name}: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text == "/quit":
                break
            if text == "/new":
                config = {"configurable": {"thread_id": f"{ctx.user_id}-{uuid.uuid4().hex[:6]}"}}
                print("   (started a new conversation thread; long-term preferences are kept)")
                continue
            if text:
                run_turn(agent, text, config, ctx)


if __name__ == "__main__":
    main()

"""
Lesson 12.2: A reflection loop: write -> grade -> revise -> grade ... (a CYCLE).
LCEL can't do this. LangGraph makes it easy.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from course_utils import banner, get_model

MIN_SCORE = 8
MAX_ATTEMPTS = 3
model = get_model(temperature=0.7)


class Grade(BaseModel):
    score: int = Field(description="Quality score from 1 (bad) to 10 (excellent)")
    feedback: str = Field(description="One specific, actionable improvement")


grader = get_model().with_structured_output(Grade, method="json_schema")


class State(TypedDict):
    topic: str
    draft: str
    score: int
    feedback: str
    attempts: int


def write(state: State) -> dict:
    if state.get("feedback"):
        prompt = (f"Rewrite this tagline for '{state['topic']}' applying the feedback.\n"
                  f"Tagline: {state['draft']}\nFeedback: {state['feedback']}\nReply with the new tagline only.")
    else:
        prompt = f"Write a catchy, one-line marketing tagline for: {state['topic']}. Reply with the tagline only."
    draft = model.invoke(prompt).content.strip().strip('"')
    attempts = state.get("attempts", 0) + 1
    print(f"  ✍️  attempt {attempts}: {draft}")
    return {"draft": draft, "attempts": attempts}


def grade(state: State) -> dict:
    g = grader.invoke([
        ("system", "You are a strict marketing editor. Score taglines harshly: 10 is rare. "
                   "Good taglines are short (under 10 words), specific and memorable."),
        ("human", f"Product: {state['topic']}\nTagline: {state['draft']}"),
    ])
    print(f"  🧑‍⚖️ score {g.score}/10: {g.feedback}")
    return {"score": g.score, "feedback": g.feedback}


def should_continue(state: State) -> str:
    if state["score"] >= MIN_SCORE:
        print("  ✅ good enough")
        return END
    if state["attempts"] >= MAX_ATTEMPTS:
        print("  ⏹️  max attempts reached")  # ALWAYS have a stop condition
        return END
    return "write"


builder = StateGraph(State)
builder.add_node("write", write)
builder.add_node("grade", grade)
builder.add_edge(START, "write")
builder.add_edge("write", "grade")
builder.add_conditional_edges("grade", should_continue, ["write", END])  # the cycle
app = builder.compile()

banner("Reflection loop")
final = app.invoke({"topic": "Sweet Crumbs' Filter Coffee Tres Leches cake"})
print(f"\nFINAL ({final['attempts']} attempts, score {final['score']}): {final['draft']}")

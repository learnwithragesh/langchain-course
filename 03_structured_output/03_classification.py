"""
Lesson 03.3: A practical use case, a support-ticket classifier.
Literal[...] restricts the model to a fixed set of labels your code can branch on.
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from course_utils import banner, get_model


class TicketClassification(BaseModel):
    """Classification of a customer support ticket."""

    category: Literal["billing", "technical", "account", "other"] = Field(
        description="Which team should handle it")
    urgency: Literal["low", "medium", "high"] = Field(
        description="high = service down or money lost, low = general question")
    sentiment: Literal["positive", "neutral", "negative"]
    summary: str = Field(description="One-sentence summary of the problem")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You triage customer support tickets for a SaaS company. Classify the ticket."),
    ("human", "Ticket:\n{ticket}"),
])
classifier = prompt | get_model().with_structured_output(TicketClassification, method="json_schema")

tickets = [
    "I was charged twice this month!! Refund me immediately.",
    "The dashboard has been showing a 500 error for everyone in our team since 9am.",
    "How do I change the email address on my profile? No rush.",
    "Just wanted to say your new feature is great, thanks!",
]

banner("Classifying tickets (batch = parallel)")
results = classifier.batch([{"ticket": t} for t in tickets])
for t, r in zip(tickets, results):
    print(f"\n  {t}")
    print(f"   -> {r.category:9} | {r.urgency:6} | {r.sentiment:8} | {r.summary}")

banner("Now ordinary Python can act on it")
urgent = [r for r in results if r.urgency == "high"]
print(f"  {len(urgent)} urgent ticket(s) -> page the on-call engineer")

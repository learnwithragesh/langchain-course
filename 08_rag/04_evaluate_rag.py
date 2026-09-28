"""
Lesson 08.4: A tiny RAG evaluation harness.

Measure RETRIEVAL and ANSWERS separately. When a score drops, you know which half to fix.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from course_utils import banner, get_bakery_store, get_model

K = 3
retriever = get_bakery_store().as_retriever(search_kwargs={"k": K})

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from the context, in one short sentence. If not in context say \"I don't know\"."
                   "\n\n{context}"),
        ("human", "{question}")])
    | get_model()
    | StrOutputParser()
)

# Each test: question, a phrase that must be in a retrieved chunk, and word(s) the answer must contain
eval_set = [
    {"q": "How many paid annual leave days do full-time staff get?", "chunk_has": "18 days", "answer_has": ["18"]},
    {"q": "What is the employee discount?", "chunk_has": "30% discount", "answer_has": ["30"]},
    {"q": "How much is delivery?", "chunk_has": "Rs 49", "answer_has": ["49"]},
    {"q": "How much notice for a wedding cake?", "chunk_has": "2 weeks", "answer_has": ["2 weeks", "two weeks"]},
    {"q": "What is the loyalty programme called?", "chunk_has": "Crumb Club", "answer_has": ["Crumb Club"]},
    {"q": "When do stores close on Tuesdays?", "chunk_has": "5:00 PM", "answer_has": ["5"]},
    {"q": "What is the prize for the Golden Whisk award?", "chunk_has": "25,000", "answer_has": ["25,000", "25000"]},
    {"q": "Who is the CEO of Sweet Crumbs in 2026?", "chunk_has": None, "answer_has": ["don't know", "do not know"]},
]

banner(f"Evaluating {len(eval_set)} questions (k={K})")
retrieval_hits = answer_hits = retrieval_total = 0
for case in eval_set:
    docs = retriever.invoke(case["q"])
    if case["chunk_has"]:
        retrieval_total += 1
        hit = any(case["chunk_has"] in d.page_content for d in docs)
        retrieval_hits += hit
    else:
        hit = None  # nothing to retrieve: tests refusal

    answer = rag_chain.invoke(case["q"])
    correct = any(a.lower() in answer.lower() for a in case["answer_has"])
    answer_hits += correct

    print(f"\nQ: {case['q']}")
    print(f"   retrieval: {'n/a' if hit is None else ('✅' if hit else '❌')}   "
          f"answer: {'✅' if correct else '❌'}  -> {answer.strip()[:100]}")

banner("Scores")
print(f"Retrieval hit rate : {retrieval_hits}/{retrieval_total} = {retrieval_hits / retrieval_total:.0%}")
print(f"Answer correctness : {answer_hits}/{len(eval_set)} = {answer_hits / len(eval_set):.0%}")
print("\nIf retrieval is low: fix chunking/k/search. If retrieval is high but answers are low: fix prompt/model.")

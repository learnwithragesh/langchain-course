# LangChain 1.x Cheatsheet

Everything important on one page. The lesson number where each topic is taught is in brackets.

## Models [L01]
```python
from langchain.chat_models import init_chat_model
model = init_chat_model("ollama:llama3.1", temperature=0)      # or "anthropic:claude-sonnet-5"
msg = model.invoke("Hi")                    # -> AIMessage; msg.content, msg.usage_metadata
for chunk in model.stream("Hi"): print(chunk.content, end="")
model.batch(["q1", "q2"])                   # parallel
await model.ainvoke("Hi")                   # async
model.invoke([("system", "Be brief."), ("user", "Hi")])
```

## Messages [L01]
```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
```

## Prompts [L02]
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
prompt.invoke({"role": "tutor", "history": [], "question": "..."})
prompt.partial(role="tutor")                # pre-fill
# literal braces: {{ }}
```

## Structured output [L03]
```python
from pydantic import BaseModel, Field
class Person(BaseModel):
    name: str = Field(description="full name")
    age: int | None = None
structured = model.with_structured_output(Person, method="json_schema")
structured.invoke("Ravi is 29")            # -> Person(name='Ravi', age=29)
# include_raw=True to debug.  Literal["a","b"] for classification.
```

## LCEL [L04]
```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, chain
from operator import itemgetter

chain_ = prompt | model | StrOutputParser()
{"a": chain1, "b": chain2}                  # dict = RunnableParallel
RunnablePassthrough.assign(x=fn)            # keep input, add key
lambda x: ...                               # function = RunnableLambda (inside a chain)
chain_.with_retry(stop_after_attempt=3)
model.with_fallbacks([backup_model])
chain_.get_graph().draw_mermaid()
```

## Memory [L05]
```python
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
trimmer = trim_messages(max_tokens=2000, strategy="last", include_system=True,
                        start_on="human", token_counter=count_tokens_approximately)
# Modern: create_agent(..., checkpointer=InMemorySaver()) + config={"configurable": {"thread_id": "x"}}
# Legacy (deprecated): RunnableWithMessageHistory, ConversationBufferMemory
```

## Documents & splitting [L06]
```python
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
Document(page_content="...", metadata={"source": "x.md"})
RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
# langchain_community loaders are SUNSET (2026): write simple loaders yourself / use pypdf etc.
```

## Embeddings & vector stores [L07]
```python
from langchain.embeddings import init_embeddings
from langchain_core.vectorstores import InMemoryVectorStore
emb = init_embeddings("ollama:nomic-embed-text")
store = InMemoryVectorStore.from_documents(chunks, embedding=emb)
store.similarity_search("q", k=4)
store.similarity_search_with_score("q", k=4)
store.max_marginal_relevance_search("q", k=4, fetch_k=20, lambda_mult=0.5)
store.similarity_search("q", filter=lambda d: d.metadata["x"] == "y")   # InMemory filter = function
retriever = store.as_retriever(search_kwargs={"k": 4})
store.dump("idx.json"); InMemoryVectorStore.load("idx.json", embedding=emb)
```

## RAG [L08]
```python
rag = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
       | prompt | model | StrOutputParser())
# Follow-ups: condense (history + question) -> standalone question -> retrieve
# Debug retrieval FIRST. Evaluate retrieval and answers separately.
```

## Tools [L09]
```python
from langchain.tools import tool, ToolRuntime
@tool
def get_price(product: str) -> str:
    """Docstring = what the model reads. Say when to use it."""
    return "60 INR"                          # include units!
get_price.invoke({"product": "bun"})
m = model.bind_tools([get_price]); m.invoke("...").tool_calls
# tool_choice="any" | "get_price"
```

## Agents [L10]
```python
from langchain.agents import create_agent
agent = create_agent(model, tools=[...], system_prompt="...",
                     checkpointer=InMemorySaver(), context_schema=Ctx,
                     response_format=Schema, middleware=[...], store=store)
out = agent.invoke({"messages": [{"role": "user", "content": "..."}]},
                   config={"configurable": {"thread_id": "t1"}}, context=Ctx(user_id="u1"))
out["messages"][-1].content;  out.get("structured_response")
for u in agent.stream(inp, stream_mode="updates"): ...     # or "messages" for tokens
# In tools: def t(x: str, runtime: ToolRuntime[Ctx]) -> runtime.context.user_id, runtime.store
```

## Middleware [L11]
```python
from langchain.agents.middleware import (SummarizationMiddleware, HumanInTheLoopMiddleware,
    ToolCallLimitMiddleware, ModelCallLimitMiddleware, PIIMiddleware, ModelFallbackMiddleware,
    before_model, after_model, wrap_model_call, wrap_tool_call, dynamic_prompt, AgentMiddleware)
SummarizationMiddleware(model=model, trigger=("tokens", 4000), keep=("messages", 20))
HumanInTheLoopMiddleware(interrupt_on={"send_email": {"allowed_decisions": ["approve", "edit", "reject"]}})
# resume: agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
@dynamic_prompt
def p(request): return f"... {request.runtime.context.user_name}"
```

## LangGraph [L12]
```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver       # SqliteSaver / PostgresSaver
from langgraph.store.memory import InMemoryStore            # SqliteStore / PostgresStore
b = StateGraph(State)
b.add_node("a", fn); b.add_edge(START, "a")
b.add_conditional_edges("a", router_fn, ["b", END])
app = b.compile(checkpointer=InMemorySaver())
app.get_state(config); app.get_state_history(config)
# Reducers: Annotated[list, operator.add], Annotated[list, add_messages]
```

## Decision guide
```
fixed steps              → LCEL chain
LLM + tools in a loop    → create_agent (+ middleware)
custom flow / cycles     → LangGraph StateGraph
```

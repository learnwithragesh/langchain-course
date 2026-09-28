# Glossary

| Term | Meaning | Lesson |
|---|---|---|
| **Agent** | An LLM that decides which tools to call, in a loop, until the task is done | 10 |
| **Agentic RAG** | RAG where retrieval is a tool the agent chooses to call, and can call repeatedly | 10 |
| **AIMessage** | A message produced by the model. It may contain `tool_calls` | 01 |
| **Checkpointer** | Saves graph or agent state after every step, keyed by `thread_id` (memory, HITL, time travel) | 05, 12 |
| **Chunk** | A piece of a document produced by a text splitter | 06 |
| **Chunk overlap** | Characters repeated between neighbouring chunks, so boundary sentences aren't lost | 06 |
| **Condensing** | Rewriting a follow-up question into a standalone one before retrieval | 08 |
| **Context (runtime)** | Data your app passes at invoke time (user ID, permissions) that tools can read but the model can't change | 10 |
| **Context window** | The maximum number of tokens a model can read in one call | 00, 05 |
| **Cosine similarity** | A measure of how closely two vectors point in the same direction (1 means identical meaning) | 07 |
| **create_agent** | LangChain 1.x's standard agent builder, built on LangGraph | 10 |
| **Document** | `page_content` + `metadata`. The unit of data in RAG | 06 |
| **Embedding** | A vector (list of numbers) representing the meaning of a text | 07 |
| **Few-shot** | Putting example input/output pairs in the prompt to teach a format | 02 |
| **Hallucination** | A confident but false model output | 00, 08 |
| **HITL** | Human-in-the-loop: pause the agent so a person approves, edits, or rejects an action | 11 |
| **Hybrid search** | Combining keyword (BM25) and vector (semantic) search | 07 |
| **Interrupt** | A LangGraph mechanism to pause execution and wait for external input | 11, 12 |
| **LangGraph** | A low-level framework for stateful, cyclic LLM workflows (graphs) | 12 |
| **LangSmith** | A hosted platform for tracing, debugging, and evaluating LLM apps | 00, 13 |
| **LCEL** | LangChain Expression Language: composing Runnables with `\|` | 04 |
| **Loader** | Turns a source (file, URL, DB) into Documents | 06 |
| **MCP** | Model Context Protocol, an open standard for exposing tools and data to LLM apps | 09 |
| **Metadata filter** | Restricting vector search to documents whose metadata matches a condition | 07 |
| **Middleware** | Hooks into the agent loop (before or after model, wrapping tools) for control and guardrails | 11 |
| **MMR** | Maximal Marginal Relevance: retrieval that balances relevance and diversity | 07 |
| **Node / Edge** | LangGraph: a node is a function step, and an edge says what runs next | 12 |
| **Output parser** | Converts model text into a Python type (str, dict, Pydantic) | 03 |
| **Prompt injection** | Malicious text (from a user, document, or tool) that tries to override instructions | 09, 10 |
| **Prompt template** | A reusable prompt with `{variables}` | 02 |
| **RAG** | Retrieval-Augmented Generation: retrieve relevant text, then generate an answer from it | 08 |
| **ReAct** | Reason + Act: the think → tool → observe loop that agents run | 09, 10 |
| **Reducer** | A LangGraph rule for merging a node's update into the state (overwrite, append, …) | 12 |
| **Retriever** | A Runnable that takes a query string and returns Documents | 07 |
| **Runnable** | LangChain's universal interface: `invoke`, `stream`, `batch` (plus async) | 01, 04 |
| **State** | The shared data dict flowing through a graph or agent (for example, `messages`) | 10, 12 |
| **Store** | Long-term, cross-thread memory (namespaced key-value storage) | 12 |
| **Streaming** | Receiving output incrementally (tokens or steps) instead of all at once | 01, 10 |
| **Structured output** | Making the model return data matching a schema | 03 |
| **System message** | Instructions that set the model's role and rules | 01 |
| **Temperature** | Output randomness (0 is the most deterministic) | 01 |
| **Thread** | One conversation, identified by `thread_id` | 05 |
| **Token** | A word piece. It's how models read text and how usage is billed | 00 |
| **Tool** | A function the model can ask your code to run (name + description + argument schema) | 09 |
| **Tool call** | The model's structured request to run a tool: `{name, args, id}` | 09 |
| **ToolMessage** | The result of a tool, sent back to the model, linked by `tool_call_id` | 09 |
| **ToolRuntime** | An injected tool parameter that gives access to context, the store, and state, hidden from the model | 10 |
| **Vector store** | A database of (vector, Document) pairs with nearest-neighbour search | 07 |

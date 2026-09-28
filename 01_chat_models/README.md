# Lesson 01: Chat Models & Messages

## Theory

### 1. Chat models vs. "LLMs"
Older LangChain had two model types: `LLM` (string in, string out) and `ChatModel` (messages in, message out).
Today **every modern model is a chat model**, so that's all you need.

A chat model takes a **list of messages** and returns **one message**.

### 2. Message types

| Class | Role | Purpose |
|---|---|---|
| `SystemMessage` | `system` | Instructions and persona ("You are a helpful tutor…"). Usually goes first. |
| `HumanMessage` | `user` | What the user says |
| `AIMessage` | `assistant` | What the model replied. It can also contain **tool calls** (L09). |
| `ToolMessage` | `tool` | The result of running a tool, sent back to the model (L09) |

Shorthand: anywhere messages are accepted, you can pass tuples such as `("system", "...")` and `("user", "...")`,
or even a plain string (which is treated as one `HumanMessage`).

### 3. Creating a model: two ways
```python
# (a) Provider class, which gives you explicit, provider-specific options
from langchain_ollama import ChatOllama
model = ChatOllama(model="llama3.1", temperature=0)

# (b) Universal factory, which is provider-agnostic (this course uses it)
from langchain.chat_models import init_chat_model
model = init_chat_model("ollama:llama3.1", temperature=0)
model = init_chat_model("anthropic:claude-sonnet-5")   # same code, different provider
```

### 4. The Runnable interface: the most important idea in LangChain
Chat models, prompts, parsers, retrievers, tools, chains, and agents all share the same methods:

| Method | What it does |
|---|---|
| `invoke(input)` | One input, one output (it blocks until done) |
| `stream(input)` | Yields output **chunks** as they are generated, the "typing" effect |
| `batch([in1, in2])` | Many inputs, processed in parallel |
| `ainvoke` / `astream` / `abatch` | Async versions for web servers |

Learn these once and you know how to call *everything*.

### 5. What comes back: `AIMessage`
- `.content`: the text
- `.response_metadata`: provider info (model name, stop reason, timing)
- `.usage_metadata`: token counts `{input_tokens, output_tokens, total_tokens}`. This is what you pay for with cloud models.
- `.tool_calls`: filled in when the model wants to call a tool (L09)

### 6. Important parameters
- **`temperature`** (0–1+): randomness. `0` gives the most predictable output, which is good for extraction and code. `0.7`+ is more creative.
- **`max_tokens`** (Ollama calls it `num_predict`): caps the length of the reply.
- **`timeout`, `max_retries`**: how the model handles slow or failed network calls with cloud providers.

### 7. Statelessness (again)
The model remembers **nothing**. To have a conversation, *you* send the full history every time:
`[system, human1, ai1, human2]`. Lesson 05 automates this.

## Scripts (run in order)
| File | Shows |
|---|---|
| `01_first_call.py` | `invoke`, and what's inside an `AIMessage` |
| `02_messages.py` | System prompts, message types, and a manual multi-turn conversation |
| `03_stream_and_batch.py` | Streaming tokens, batching, and async |
| `04_parameters.py` | How temperature and max tokens change the output |

## Pitfalls
- Forgetting that `.invoke()` returns an `AIMessage` rather than a string. Use `.content`.
- Expecting the model to remember a previous `invoke`. It won't.
- A high temperature for tasks that need exact answers.

## Exercises
1. Change the system prompt in `02_messages.py` to "Answer like a pirate". What changes?
2. In `02_messages.py`, remove the earlier AI message from history. Can the model still answer the follow-up?
3. Run `04_parameters.py` twice. Which outputs changed between runs and which stayed the same? Why?
4. If you have an API key, set `LC_CHAT_MODEL=anthropic:claude-sonnet-5` and compare `usage_metadata`.

# Lesson 02: Prompt Templates

## Theory

### 1. Why templates?
Hard-coded f-strings work for one script, but real apps need prompts that are:
- **reusable**: the same structure with different inputs
- **validated**: an error if you forget a variable
- **composable**: they plug into chains with `|` (L04)
- **message-aware**: they produce a list of system, human, and AI messages, not just one string

A **prompt template** is a Runnable that takes a `dict` of variables and returns a **PromptValue**
(something a chat model can accept directly).

```
{"topic": "black holes", "level": "beginner"}
          │
          ▼
  ChatPromptTemplate  ──▶  [SystemMessage("You teach beginner students"),
                            HumanMessage("Explain black holes")]
```

### 2. The main classes

| Class | Output | Use when |
|---|---|---|
| `PromptTemplate` | a single string | Simple text prompts or embedding a snippet |
| `ChatPromptTemplate` | a list of messages | **Almost always.** Chat models want messages. |
| `MessagesPlaceholder` | slot for a *list* of messages | Chat history (L05) and agent scratchpads |
| `FewShotChatMessagePromptTemplate` | example input/output pairs | Teaching the model a format by example |

### 3. Syntax
```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])
prompt.invoke({"role": "chef", "question": "How do I boil an egg?"})
```
- Variables use `{curly}` braces. Write a literal brace as `{{` or `}}`, which matters when your prompt contains JSON.
- `prompt.input_variables` tells you which variables are required.
- `.partial(role="chef")` pre-fills some variables and returns a new template.

### 4. Prompt engineering basics (these apply to every model)
1. **Role**: say who the model is ("You are a senior Python reviewer").
2. **Task**: be specific about what to do.
3. **Context**: give the data it needs; don't make it guess.
4. **Format**: say exactly what the output should look like (length, bullets, JSON).
5. **Examples** (few-shot): show 2–3 input→output pairs. This is the most reliable way to get a format.
6. **Constraints**: say what *not* to do ("If unsure, say 'I don't know'").

Small local models (like llama3.1 8B) need these rules *much* more than large models.

## Scripts
| File | Shows |
|---|---|
| `01_templates.py` | `PromptTemplate`, `ChatPromptTemplate`, variables, `partial`, and inspecting output |
| `02_template_with_model.py` | Your first chain: `prompt \| model \| parser` |
| `03_few_shot.py` | Few-shot prompting to enforce an output style |
| `04_messages_placeholder.py` | Inserting a whole list of messages (a preview of memory) |

## Pitfalls
- JSON inside a template: `{"a": 1}` breaks because `{"a"...}` is read as a variable. Use `{{"a": 1}}`.
- Forgetting a variable raises a `KeyError` at invoke time. Check `prompt.input_variables`.
- Putting everything in the human message. Stable instructions belong in the **system** message.

## Exercises
1. Build a template that takes `{language}` and `{code}` and asks the model to explain the code. Run it on a snippet.
2. Add a third few-shot example in `03_few_shot.py` that changes the style. Does the output follow it?
3. Use `.partial()` to make a "Tamil tutor" version of a generic tutor template.

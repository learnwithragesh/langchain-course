# Lesson 03: Structured Output

## Theory

### 1. The problem
LLMs return **text**. Your program needs **data**: a `dict`, an `int`, a list of objects, or a category label.
Parsing free text with regex breaks the moment the model phrases things differently.

```
"Ravi, 29, from Chennai, loves chess"   ──▶   Person(name="Ravi", age=29, city="Chennai", hobbies=["chess"])
```

### 2. Two generations of solutions

**(a) Output parsers (older and portable).** You put *format instructions* in the prompt ("Reply as JSON with keys…"),
then parse the text afterwards.
- `StrOutputParser`: `AIMessage` → `str`
- `JsonOutputParser`: text → `dict`
- `PydanticOutputParser`: text → a validated Pydantic object

The weakness is that the model can still ignore the instructions. It's a request, not a guarantee.

**(b) `model.with_structured_output(Schema)` (modern and recommended).**
You pass the schema to the model API itself, using the provider's native "JSON mode", "structured outputs", or "tool calling".
Many providers then **constrain decoding**, so the model *cannot* produce invalid JSON.
LangChain validates the result and returns a Python object.

```python
class Person(BaseModel):
    name: str = Field(description="full name")
    age: int

structured = model.with_structured_output(Person, method="json_schema")
structured.invoke("Ravi is 29")      # -> Person(name='Ravi', age=29)
```

### 3. Schema options
| Schema type | You get back |
|---|---|
| Pydantic `BaseModel` (recommended) | A validated Pydantic instance (with type coercion and errors) |
| `TypedDict` | A plain `dict` |
| JSON Schema `dict` | A plain `dict` |

### 4. `method=`
- `"json_schema"`: uses the provider's native structured-output or JSON mode. It works well on Ollama and Claude, so this course uses it.
- `"function_calling"`: pretends the schema is a tool and forces the model to "call" it.
- `"json_mode"`: plain JSON mode without a schema (older OpenAI).

### 5. How to make it work well
1. **Field descriptions matter.** They're sent to the model and act as instructions. `Field(description="age in years, integer")`.
2. **Docstrings on the class** describe the overall object.
3. Use **`Literal["pos","neg","neutral"]`** or `Enum` for classification. This restricts the answer to allowed labels.
4. Use **`Optional[...] = None`** for fields that might not be in the text. Otherwise the model invents values (hallucinates).
5. Use **`list[SubModel]`** to extract many items at once.
6. Add a **system message** that explains the task. Small models need it.
7. `include_raw=True` returns `{"raw": AIMessage, "parsed": obj, "parsing_error": err}`, which is great for debugging.

### 6. Where this is used later
- Routing (L04): the model classifies a question, then code branches on the label.
- Agents (L10): `create_agent(..., response_format=Schema)` returns structured final answers.
- Extraction pipelines: invoices, resumes, and emails into database rows.

## Scripts
| File | Shows |
|---|---|
| `01_output_parsers.py` | `StrOutputParser`, `JsonOutputParser`, and format instructions |
| `02_with_structured_output.py` | Pydantic extraction, nested lists, `Optional`, and `include_raw` |
| `03_classification.py` | A practical support-ticket classifier using `Literal` |

## What we observed with llama3.1 (8B)
On a real run of `02_with_structured_output.py`:
- For "I'm Ravi Kumar, 29, from Chennai…", it returned `city=None`. The model **missed** a stated fact.
- For "Meena works as a designer.", it returned `hobbies=['']`, which is an empty-string hobby instead of an empty list.

The schema guarantees the **shape** of the output. It does **not** guarantee that the **content** is correct.
Always validate important fields, and use a stronger model or a clearer prompt when accuracy matters.

## Pitfalls
- No `Optional` means the model invents a value for missing info.
- Vague field names like `x` or `data`: the name *is* part of the prompt.
- Very large schemas on small models. Split them into several smaller extractions.

## Exercises
1. Write an `Invoice` schema (vendor, date, total, a list of line items) and extract it from a made-up invoice text.
2. Add a `confidence: float` field to the classifier. Is the model's confidence meaningful?
3. Remove the `Optional` from `email` in script 02 and run it on text with no email. What does the model return?

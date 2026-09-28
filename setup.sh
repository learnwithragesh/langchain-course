#!/usr/bin/env bash
# One-time setup for the LangChain course. Safe to re-run.
set -e
cd "$(dirname "$0")"

# 1. Python 3.12 virtual environment (3.12 has the widest package support)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Make `import course_utils` work from every lesson folder
echo "$PWD" > .venv/lib/python3.12/site-packages/langchain_course.pth

# 3. Local models (free, offline)
ollama pull llama3.1
ollama pull nomic-embed-text

echo "Done. Activate with:  source .venv/bin/activate"

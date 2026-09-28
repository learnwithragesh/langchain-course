#!/usr/bin/env bash
# Runs every non-interactive lesson script and reports pass/fail.
# Use it after upgrading packages or switching models:  ./run_all_lessons.sh
cd "$(dirname "$0")"
source .venv/bin/activate
out=$(mktemp)
pass=0; fail=0
for f in 0*/*.py 1[0-2]*/*.py; do
  [[ "$f" == *chatbot.py ]] && continue            # interactive
  if python "$f" > "$out" 2>&1; then
    echo "✅ $f"; pass=$((pass+1))
  else
    echo "❌ $f"; tail -5 "$out" | sed 's/^/     /'; fail=$((fail+1))
  fi
done
if python 13_capstone_project/app.py --demo > "$out" 2>&1; then
  echo "✅ 13_capstone_project/app.py --demo"; pass=$((pass+1))
else
  echo "❌ 13_capstone_project/app.py --demo"; fail=$((fail+1))
fi
echo "---- $pass passed, $fail failed ----"

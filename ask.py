"""
Ask the assistant from the terminal.

    uv run ask.py                                     the demo questions
    uv run ask.py "how long does a refund take"       as guest
    uv run ask.py "what does ERR-4010 mean" alice     as Alice

People: alice (engineering), bob (hr), carol (finance), guest (outside).
"""

import sys

from assistant.pipeline import Assistant

sys.stdout.reconfigure(encoding="utf-8")

# The same question asked by two roles, twice, then one nobody can answer.
DEMO = [
    ("How long does a refund take?", "guest"),
    ("What does our compensation adjustment policy say?", "alice"),
    ("What does our compensation adjustment policy say?", "bob"),
    ("What is the current meal reimbursement limit?", "carol"),
    ("What was the previous meal reimbursement limit?", "carol"),
    ("What is our policy on cryptocurrency payments?", "bob"),
]


def ask(assistant, question, username):
    answer = assistant.ask(question, username=username)

    print(f"[{username}] {question}")
    print(f"  {answer.text}")

    for citation in answer.citations:
        mark = "verified" if citation.grounded else "NOT VERIFIED"
        print(f"  {mark}: {citation.label} (reviewed {citation.updated_on})")

    for warning in answer.warnings:
        print(f"  warning: {warning}")

    print()


assistant = Assistant()

if len(sys.argv) > 1:
    question = sys.argv[1]
    username = sys.argv[2] if len(sys.argv) > 2 else "guest"
    ask(assistant, question, username)
else:
    for question, username in DEMO:
        ask(assistant, question, username)

"""parsantic end-to-end demo — real LLM extraction from unstructured text.

Requires: uv sync --extra ai
Requires: GEMINI_API_KEY env var by default,
or set PARSANTIC_MODEL to use another provider/model.
"""

import os

from pydantic import BaseModel

import parsantic as sap
from parsantic.extract import Prompt

# ── Define what we want to extract ───────────────────────────────────


class Person(BaseModel):
    name: str
    role: str
    years_experience: int | None = None


# ── Unstructured text (imagine a resume, bio, or article) ────────────

text = """
Dr. Sarah Chen is a principal machine learning engineer at Anthropic,
where she has worked for the past 3 years. Before that, she spent 5 years
at Google Brain working on large language models. She holds a PhD in
computer science from Stanford University. In her spare time, she
mentors junior engineers and contributes to open-source projects.
"""

# ── One line: text + schema → typed object ───────────────────────────

model = os.getenv("PARSANTIC_MODEL", "gemini:gemini-2.5-flash")

result = sap.extract(text, Person, model=model)
print("extract() result:")
print(f"  {result.value!r}")
print(f"  flags={result.flags}  score={result.score}")
print()

# ── Same thing with a custom prompt ──────────────────────────────────

bio2 = """
Marcus Johnson, a 28-year-old software developer from Austin, Texas,
recently joined Stripe as a backend engineer. He previously worked at
Shopify for two years. Marcus is passionate about distributed systems
and has been coding professionally for 6 years total.
"""

result2 = sap.extract(
    bio2,
    Person,
    model=model,
    prompt=Prompt(description="Extract the person's professional details."),
)
print("extract() with custom prompt:")
print(f"  {result2.value!r}")

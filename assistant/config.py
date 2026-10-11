"""
Every setting in one place.

If a number or a name matters anywhere in this project, it is here. Nothing
else in the code contains a magic value, because the first thing you do with a
RAG system is tune it, and hunting values across six files wastes an afternoon.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / "sample-documents"
PROMPTS_DIR = ROOT / "prompts"
SCHEMA_FILE = ROOT / "schema.sql"

# The database. One Postgres holds the documents, the vectors and the access
# rules, so this single string is the whole data layer.
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://rag:ragsecret@localhost:5433/ragdb"
)

# Models. Changing EMBEDDING_MODEL means re-indexing everything, because vectors
# from two different models cannot be compared.
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"
HELPER_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# Retrieval
CANDIDATES = 12  # how many each search brings back before reranking
TOP_K = 4  # how many survive and reach the model
MIN_RERANK_SCORE = 4  # 0 to 10, below this a chunk is dropped
HIGH_CONFIDENCE_SCORE = 8  # at or above this the reranker is trusted without a second opinion
RRF_CONSTANT = 60  # the constant in reciprocal rank fusion

# The self correcting loop
MAX_ATTEMPTS = 2  # one retry with a rewritten query, then give up honestly

# Freshness. A document older than this is still used, but the answer says so.
STALE_AFTER_DAYS = 180

# Access control, as one rule:
#
#   public      anyone, including people outside the company
#   internal    any employee, whatever their department
#   restricted  only the department that owns the document
#
# It is applied in SQL before the search runs, so a document a person may not
# read is never a candidate, never reranked and never seen by the model. The
# users themselves live in the database, seeded by schema.sql, because in a
# real company they come from the identity provider.


def prompt(name: str) -> str:
    """Read a prompt template from the prompts folder.

    Prompts live in files, not in Python strings, so a non developer can read
    and edit them without touching the code.
    """
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")

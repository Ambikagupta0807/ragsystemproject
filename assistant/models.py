"""
The shapes that travel between the layers.

Plain dataclasses, no behaviour. Reading this file tells you what the system
moves around: a user asks, documents are chunked, chunks come back as hits, and
an answer carries its citations and its trace.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class User:
    """Who is asking. Loaded from the database, never from the question."""

    username: str
    display_name: str
    department: str
    is_employee: bool

    def describe_access(self) -> str:
        if not self.is_employee:
            return "public documents only"
        return f"public and internal documents, plus restricted {self.department} documents"


@dataclass
class SourceDocument:
    """One file from the documents folder, with the header it declares."""

    source: str
    title: str
    department: str
    visibility: str
    version: int
    status: str
    effective_from: date
    updated_on: date
    content_hash: str
    body: str


@dataclass
class Chunk:
    """A piece of a document, ready to be embedded and stored."""

    section: str
    content: str


@dataclass
class Hit:
    """A chunk that came back from search, with where it came from."""

    chunk_id: int
    content: str
    section: str
    source: str
    title: str
    version: int
    status: str
    updated_on: date
    score: float = 0.0  # fusion score first, then the reranker score

    @property
    def label(self) -> str:
        """How this chunk is cited: title, version and section."""
        return f"{self.title} v{self.version} > {self.section}"


@dataclass
class Citation:
    """One claim in the answer, tied back to the chunk that supports it."""

    label: str
    quote: str
    updated_on: date
    status: str
    grounded: bool


@dataclass
class Answer:
    """What the pipeline returns. Everything the caller needs, nothing hidden."""

    question: str
    text: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)

    @property
    def fully_grounded(self) -> bool:
        return bool(self.citations) and all(c.grounded for c in self.citations)

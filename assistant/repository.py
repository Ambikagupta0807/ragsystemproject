"""
Every line of SQL in this project lives here.

The rest of the code asks this layer for users, documents and chunks, and never
writes a query of its own. That is the same reason a Spring project keeps its
repository package separate: when something is wrong in the database, there is
exactly one file to open.

The interesting query is hybrid_search. It does five things in one round trip:

    1. keeps only documents this person is allowed to read
    2. keeps only the current version, unless history was asked for
    3. ranks chunks by vector distance
    4. ranks the same chunks by keyword match
    5. merges the two rankings with reciprocal rank fusion

Postgres does all of it, so there is no second system to keep in sync and no
merging logic in Python.
"""

import atexit
from contextlib import contextmanager

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from assistant import config
from assistant.models import Hit, SourceDocument, User

_pool: ConnectionPool | None = None


def _create_extension() -> None:
    """Create the vector extension once, before the pool opens.

    It cannot go in _prepare. The pool opens several connections at the same
    time, and CREATE EXTENSION IF NOT EXISTS run concurrently does not quietly
    do nothing, it raises a duplicate key error on pg_extension.
    """
    with psycopg.connect(config.DATABASE_URL) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()


def _prepare(connection: psycopg.Connection) -> None:
    """Run on every new connection in the pool.

    register_vector looks the type up in the database, which is why the
    extension has to exist before any pooled connection is opened.
    """
    register_vector(connection)


def pool() -> ConnectionPool:
    """One pool for the process.

    A single long lived connection looks simpler until the database restarts,
    and then every request fails with "the connection is closed" until the app
    itself is restarted. The pool checks a connection before handing it over and
    opens a new one when the old one is gone, so a database restart costs one
    slow request instead of an outage.
    """
    global _pool

    if _pool is None:
        _create_extension()
        _pool = ConnectionPool(
            config.DATABASE_URL,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
            configure=_prepare,
            check=ConnectionPool.check_connection,
            open=True,
        )
        # Scripts exit when their work is done, and an open pool would print a
        # warning about the thread it still owns.
        atexit.register(close_pool)

    return _pool


def close_pool() -> None:
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection():
    """Borrow a connection for one unit of work, then give it back."""
    with pool().connection() as borrowed:
        yield borrowed


def create_schema(connection: psycopg.Connection) -> None:
    """Run schema.sql. Safe to call every time, everything is IF NOT EXISTS."""
    connection.execute(config.SCHEMA_FILE.read_text(encoding="utf-8"))
    connection.commit()


# -------------------------------------------------------------------- users


def find_user(connection: psycopg.Connection, username: str) -> User:
    """Who is asking. An unknown name gets the least access, never an error."""
    row = connection.execute(
        "SELECT username, display_name, department, is_employee FROM users WHERE username = %s",
        (username,),
    ).fetchone()

    if row is None:
        return User(
            username=username, display_name=username, department="none", is_employee=False
        )

    return User(**row)


# ---------------------------------------------------------------- documents


def find_by_hash(connection: psycopg.Connection, content_hash: str) -> dict | None:
    """Has this exact text been indexed already, under any file name?"""
    row = connection.execute(
        "SELECT id, source, title, version FROM documents WHERE content_hash = %s",
        (content_hash,),
    ).fetchone()
    return dict(row) if row else None


def find_by_version(connection: psycopg.Connection, title: str, version: int) -> dict | None:
    """The row this upload would replace: the same policy, the same version."""
    row = connection.execute(
        "SELECT id, source, content_hash FROM documents WHERE title = %s AND version = %s",
        (title, version),
    ).fetchone()
    return dict(row) if row else None


def delete_document(connection: psycopg.Connection, document_id: int) -> None:
    """Remove a document and its chunks, which cascade."""
    connection.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def list_documents(connection: psycopg.Connection) -> list[dict]:
    """What is indexed right now. Shown on the upload screen."""
    rows = connection.execute(
        """
        SELECT d.title, d.version, d.status, d.department, d.visibility,
               d.source, d.updated_on, count(c.id) AS chunks
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.title, d.version
        """
    ).fetchall()
    return [dict(row) for row in rows]


def insert_document(connection: psycopg.Connection, document: SourceDocument) -> int:
    row = connection.execute(
        """
        INSERT INTO documents
            (source, title, department, visibility, version, status,
             effective_from, updated_on, content_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            document.source,
            document.title,
            document.department,
            document.visibility,
            document.version,
            document.status,
            document.effective_from,
            document.updated_on,
            document.content_hash,
        ),
    ).fetchone()
    return row["id"]


def insert_chunks(
    connection: psycopg.Connection,
    document_id: int,
    sections: list[str],
    contents: list[str],
    embeddings: list[list[float]],
) -> None:
    """Store every chunk of one document in a single statement."""
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (document_id, section, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            list(zip([document_id] * len(contents), sections, contents, embeddings)),
        )


def counts(connection: psycopg.Connection) -> dict:
    """Used by the health endpoint and by the ingest script."""
    row = connection.execute(
        """
        SELECT (SELECT count(*) FROM documents)                         AS documents,
               (SELECT count(*) FROM documents WHERE status = 'active') AS active_documents,
               (SELECT count(*) FROM chunks)                            AS chunks,
               (SELECT max(indexed_at) FROM documents)                  AS last_indexed
        """
    ).fetchone()
    return dict(row)


# ------------------------------------------------------------------- search

# One CTE per idea, so the query reads like the pipeline it implements.
HYBRID_SEARCH = """
WITH readable AS (
    SELECT id FROM documents
    WHERE (
        visibility = 'public'
        OR (visibility = 'internal'   AND %(is_employee)s)
        OR (visibility = 'restricted' AND department = %(department)s)
    )
    AND (status = 'active' OR %(include_history)s)
),
semantic AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(query_vector)s) AS rank
    FROM chunks c
    JOIN readable r ON r.id = c.document_id
    ORDER BY c.embedding <=> %(query_vector)s
    LIMIT %(candidates)s
),
keyword AS (
    SELECT c.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank(c.search_text, query) DESC) AS rank
    FROM chunks c
    JOIN readable r ON r.id = c.document_id,
         plainto_tsquery('english', %(question)s) AS query
    WHERE c.search_text @@ query
    ORDER BY ts_rank(c.search_text, query) DESC
    LIMIT %(candidates)s
),
fused AS (
    SELECT id, SUM(1.0 / (%(rrf_constant)s + rank)) AS score
    FROM (SELECT * FROM semantic UNION ALL SELECT * FROM keyword) AS ranked
    GROUP BY id
    ORDER BY score DESC
    LIMIT %(candidates)s
)
SELECT c.id AS chunk_id, c.content, c.section,
       d.source, d.title, d.version, d.status, d.updated_on,
       f.score
FROM fused f
JOIN chunks c    ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
"""


def hybrid_search(
    connection: psycopg.Connection,
    question: str,
    query_vector: list[float],
    user: User,
    include_history: bool = False,
) -> list[Hit]:
    """Meaning and keywords, merged by rank, inside this person's permissions."""
    rows = connection.execute(
        HYBRID_SEARCH,
        {
            "question": question,
            # Wrapped so Postgres receives a vector, not a plain float array.
            "query_vector": Vector(query_vector),
            "is_employee": user.is_employee,
            "department": user.department,
            "include_history": include_history,
            "candidates": config.CANDIDATES,
            "rrf_constant": config.RRF_CONSTANT,
        },
    ).fetchall()

    return [
        Hit(
            chunk_id=row["chunk_id"],
            content=row["content"],
            section=row["section"],
            source=row["source"],
            title=row["title"],
            version=row["version"],
            status=row["status"],
            updated_on=row["updated_on"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def searchable_chunks(
    connection: psycopg.Connection, user: User, include_history: bool = False
) -> dict:
    """How much of the library this person could search at all.

    This goes into the trace, because "9 of 44 chunks were never candidates for
    you" says more about access control than any sentence on a slide.
    """
    row = connection.execute(
        """
        SELECT count(*) FILTER (
            WHERE (
                d.visibility = 'public'
                OR (d.visibility = 'internal'   AND %(is_employee)s)
                OR (d.visibility = 'restricted' AND d.department = %(department)s)
            )
            AND (d.status = 'active' OR %(include_history)s)
        ) AS allowed,
        count(*) AS total
        FROM chunks c JOIN documents d ON d.id = c.document_id
        """,
        {
            "is_employee": user.is_employee,
            "department": user.department,
            "include_history": include_history,
        },
    ).fetchone()
    return dict(row)


# -------------------------------------------------------------------- audit


def record_event(
    connection: psycopg.Connection,
    username: str,
    question: str,
    answered: bool,
    grounded: bool,
    sources: str,
) -> None:
    """One row per question. An assistant over private documents needs this."""
    connection.execute(
        """
        INSERT INTO audit_events (username, question, answered, grounded, sources)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (username, question, answered, grounded, sources),
    )
    connection.commit()


def recent_events(connection: psycopg.Connection, limit: int = 10) -> list[dict]:
    rows = connection.execute(
        "SELECT at, username, question, answered, grounded, sources "
        "FROM audit_events ORDER BY at DESC LIMIT %s",
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]

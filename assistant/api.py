"""
The HTTP layer. Four endpoints, no business logic.

    GET    /             the web page
    GET    /health       is the database up, and what is in it
    GET    /documents    what has been uploaded so far
    POST   /documents    upload a document, which indexes it
    POST   /ask          {"question": "...", "username": "alice"}

The user arrives here from the request because this is a class project. In a
real deployment it comes from the session or the access token, and nothing
below this file changes.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from assistant import config, ingestion, repository
from assistant.pipeline import Assistant

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Open the pool and make sure the tables exist, once."""
    with repository.connection() as connection:
        repository.create_schema(connection)

    state["assistant"] = Assistant()
    state["started_at"] = datetime.now(timezone.utc)
    yield
    repository.pool().close()


app = FastAPI(title="Secure Enterprise RAG Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    username: str = "guest"


@app.get("/health")
def health() -> dict:
    with repository.connection() as connection:
        # started_at tells you whether you are talking to the server you just
        # started, or to an older one still holding the port.
        return {
            "status": "ok",
            "started_at": state["started_at"].isoformat(timespec="seconds"),
            **repository.counts(connection),
        }


@app.get("/documents")
def documents() -> dict:
    with repository.connection() as connection:
        return {"documents": repository.list_documents(connection)}


@app.post("/documents")
async def upload(file: UploadFile) -> dict:
    text = (await file.read()).decode("utf-8", errors="replace")

    with repository.connection() as connection:
        try:
            result = ingestion.store(connection, text, file.filename)
        except ingestion.UploadError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return {
            "outcome": result.outcome,
            "message": result.line(),
            **repository.counts(connection),
        }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    answer = state["assistant"].ask(request.question, username=request.username)

    return {
        "question": answer.question,
        "answer": answer.text,
        "answered": answer.answered,
        "grounded": answer.fully_grounded,
        "citations": [
            {
                "source": citation.label,
                "quote": citation.quote,
                "updated_on": str(citation.updated_on),
                "status": citation.status,
                "grounded": citation.grounded,
            }
            for citation in answer.citations
        ],
        "warnings": answer.warnings,
        "trace": answer.trace,
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(config.ROOT / "web" / "index.html")

# uv run python -c "import fastapi, langchain_openai, psycopg, pgvector; print('all good')"
# uv run uvicorn assistant.api:app --reload
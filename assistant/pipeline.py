"""
The pipeline. This file is the whole system in one page.

    who is asking   load the user and their department from the database
    current or old  does the question want today's policy or an older one
    retrieve        hybrid search inside their permissions, then rerank
    judge           are these chunks enough to answer
    correct         if not, rewrite the question and search once more
    answer          write it, cite it, verify every quote
    record          one audit row per question

Every step appends a line to a trace, which the API returns and the web page
shows. When this assistant surprises somebody in production, nobody has to
guess what it did.
"""

from datetime import date

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from assistant import config, repository
from assistant.generation import Generator
from assistant.models import Answer
from assistant.retrieval import Retriever

#retrieval grading result
 #After retrieval, we need to decide:
#
# "Do these chunks contain enough information
# to answer the user's question?"
class _Grade(BaseModel):
    sufficient: bool = Field(description="True only if these chunks contain the answer")
    missing: str = Field(description="What is missing, or 'nothing'")


class _Intent(BaseModel):
    wants_history: bool = Field(
        description="True when the question asks about an older or previous version of a rule"
    )


class Assistant:
    """The application service. Build it once, ask it many questions."""

    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.helper = ChatOpenAI(model=config.HELPER_MODEL, temperature=0)
        self.grader = self.helper.with_structured_output(_Grade)
        self.intent = self.helper.with_structured_output(_Intent)

    def ask(self, question: str, username: str = "guest", today: date | None = None) -> Answer:
        """One question, one borrowed connection, returned when the answer is ready."""
        with repository.connection() as connection:
            return self._ask(connection, question, username, today)


        #start of the RAG flow

    def _ask(self, connection, question: str, username: str, today: date | None) -> Answer:
        # idenity the user
        user = repository.find_user(connection, username)


        # Most questions are about the rule that applies today, so superseded
        # versions stay out of the search entirely. A question asking what the
        # rule used to be needs the opposite, so we decide before searching.
        wants_history = self.intent.invoke(
            config.prompt("intent") + "\n\nQuestion: " + question
        ).wants_history


        # check searchabel doc
    #asking postgree howmany chunks exist and how many can this user search

        reach = repository.searchable_chunks(connection, user, wants_history)

        trace = [
            f"user {user.display_name} ({user.department}), can read {user.describe_access()}",
            f"searchable for this user: {reach['allowed']} of {reach['total']} chunks",
            f"history requested: {wants_history}",
        ]

        answer = self._attempt(connection, question, user, wants_history, trace, today)
        self._record(connection, user, question, answer)
        return answer

    #corrective rag

    def _attempt(self, connection, question, user, wants_history, trace, today) -> Answer:
        query = question

        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            hits = self.retriever.search(connection, query, user, wants_history)
            trace.append(f"attempt {attempt}: searched '{query}', kept {len(hits)} chunks")

            # When the reranker is this sure, asking a second model to agree
            # costs a call and only adds a way to disagree with ourselves.
            if hits and hits[0].score >= config.HIGH_CONFIDENCE_SCORE:
                trace.append(f"attempt {attempt}: top chunk scored {hits[0].score}/10, answering")
                return self._answer(question, hits, trace, today)

            if hits:
                grade = self.grader.invoke(
                    config.prompt("grade").format(
                        question=question,
                        context="\n\n".join(f"[{hit.label}] {hit.content}" for hit in hits),
                    )
                )
                trace.append(f"attempt {attempt}: sufficient={grade.sufficient}")

                if grade.sufficient:
                    return self._answer(question, hits, trace, today)
            else:
                grade = _Grade(sufficient=False, missing="nothing relevant was found")

            if attempt < config.MAX_ATTEMPTS:
                query = self.helper.invoke(
                    config.prompt("rewrite").format(question=question, missing=grade.missing)
                ).content.strip()
                trace.append(f"corrective retrieval: rewrote the query to '{query}'")

        trace.append("gave up and escalated")
        return Answer(
            question=question,
            text=(
                "I could not find a reliable answer in the documents you are allowed to read. "
                "I have not guessed. Please raise a ticket and a person will pick this up."
            ),
            answered=False,
            warnings=["Escalated to a human after two attempts."],
            trace=trace,
        )

    def _answer(self, question, hits, trace, today) -> Answer:
        answer = self.generator.write(question, hits, today=today)
        trace.append(
            f"answered with {len(answer.citations)} citation(s), grounded={answer.fully_grounded}"
        )
        answer.trace = trace
        return answer

    def _record(self, connection, user, question, answer: Answer) -> None:
        """One row per question, so the whole thing is auditable."""
        repository.record_event(
            connection,
            username=user.username,
            question=question,
            answered=answer.answered,
            grounded=answer.fully_grounded,
            sources=", ".join(citation.label for citation in answer.citations) or "none",
        )

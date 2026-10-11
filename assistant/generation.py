"""
Generation, with the model kept on a short leash.

The model is allowed to do one thing: phrase an answer out of the chunks it was
handed. It may not add knowledge of its own, and for every claim it must point
at a chunk and copy the sentence that supports it.

Then we check that sentence ourselves, in Python, against the chunk. Asking a
model whether it told the truth is weak. Comparing a quoted string against its
source is not, and it costs nothing.

An answer also carries its freshness. If the newest document behind it was
reviewed a long time ago, the answer says so rather than pretending the policy
is current.
"""

from datetime import date

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from assistant import config
from assistant.models import Answer, Citation, Hit

# We want LLM to tell us ;
#which chunks supports the asnwer
#what exact sentence supports it?
class _DraftCitation(BaseModel):
    chunk_number: int = Field(description="Number of the chunk this claim came from")
    quote: str = Field(description="Exact sentence from that chunk, copied word for word")


# {
#     "answer":"rag is......",
#     "citations":[
#         {
#             "chunk_number":
#         }
#     ]
# }
class _Draft(BaseModel):
    answer: str = Field(description="The answer, four lines at most")
    citations: list[_DraftCitation] = Field(description="One per claim, empty if you cannot answer")
    answered: bool = Field(description="False when the chunks do not contain the answer")


# Added to the prompt only on a second attempt, when a quote from the first one
# could not be found in its document.
REMINDER = (
    "\n\nYour previous attempt quoted a sentence that does not appear in any "
    "document. Find the sentence in the document and copy it character for "
    "character, including its first and last words."
)


def _flatten(text: str) -> str:
    return " ".join(text.split()).lower()


class Generator:
    """Writes the answer and proves it."""

    def __init__(self):
        self.model = ChatOpenAI(
            model=config.ANSWER_MODEL, temperature=0
        ).with_structured_output(_Draft)

    def write(self, question: str, hits: list[Hit], today: date | None = None) -> Answer:
        """Write the answer, and try once more if a quote does not check out."""
        answer = self._draft(question, hits, today or date.today())

        if answer.citations and not answer.fully_grounded:
            answer = self._draft(question, hits, today or date.today(), reminder=REMINDER)

        return answer

    def _draft(
        self,
        question: str,
        hits: list[Hit],
        today: date,
        reminder: str = "",
    ) -> Answer:
        if not hits:
            return Answer(
                question=question,
                text="I could not find this in the documents I am allowed to read.",
                answered=False,
            )

        context = "\n\n".join(
            f"[{number}] (source: {hit.label}, {hit.status}, reviewed {hit.updated_on})"
            f"\n{hit.content}"
            for number, hit in enumerate(hits, start=1)
        )

        draft = self.model.invoke([
            {
                "role": "system",
                "content": config.prompt("answer").format(context=context) + reminder,
            },
            {"role": "user", "content": question},
        ])

        citations = []
        for item in draft.citations:
            if not 1 <= item.chunk_number <= len(hits):
                continue

            hit = hits[item.chunk_number - 1]
            citations.append(
                Citation(
                    label=hit.label,
                    quote=item.quote,
                    updated_on=hit.updated_on,
                    status=hit.status,
                    grounded=_flatten(item.quote) in _flatten(hit.content),
                )
            )

        answer = Answer(
            question=question,
            text=draft.answer,
            answered=draft.answered,
            citations=citations,
        )

        self._add_warnings(answer, hits, today)
        return answer

    def _add_warnings(self, answer: Answer, hits: list[Hit], today: date) -> None:
        if answer.citations and not answer.fully_grounded:
            answer.warnings.append(
                "A quoted sentence was not found in its source, so this answer is not trusted."
            )

        age_in_days = (today - max(hit.updated_on for hit in hits)).days
        if age_in_days > config.STALE_AFTER_DAYS:
            answer.warnings.append(
                f"The newest document behind this answer was reviewed {age_in_days} days ago."
            )

        if any(citation.status == "superseded" for citation in answer.citations):
            answer.warnings.append(
                "This quotes a superseded version. It is history, not current policy."
            )

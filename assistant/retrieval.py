"""
Retrieval, the half of RAG that runs per question.

    embed the question  ->  hybrid search in Postgres  ->  rerank  ->  cut

The search itself is one SQL statement in the repository, and it already does
access control and rank fusion. This layer adds the second pass.

Why rerank at all: a chunk was embedded long before anyone asked a question, so
its vector has to represent that chunk for every possible question at once. The
reranker sees the question and the chunk together, which is a much easier
judgement, and it is the cheapest large quality gain in a RAG system.

Why cut below a score: taking the best four blindly would pass a chunk the
reranker called useless straight into the prompt, which is exactly what we were
trying to avoid.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from assistant import config, repository
from assistant.models import Hit, User


#1 rag is best
#2 rag is tough
#3 rag system are very useful

# {
#     "index":3
#     "score": 8
# }

class _Rating(BaseModel):
    index: int = Field(description="Number of the chunk being rated")
    score: int = Field(description="0 means useless, 10 means it fully answers the question")


#structure of multiple ratings as llms needs to return multiple ratings
class _Ratings(BaseModel):
    ratings: list[_Rating] = Field(description="One rating per candidate")


# complete retrieval process

class Retriever:
    """Hybrid search with reranking, inside one caller's permissions."""

#conver users question into embedding vector
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.reranker = ChatOpenAI(
            model=config.HELPER_MODEL, temperature=0
        ).with_structured_output(_Ratings)

#
    def search(
        self, connection, question: str, user: User, include_history: bool = False
    ) -> list[Hit]:
        """The connection is passed in, so it can come from the pool per request."""
        query_vector = self.embeddings.embed_query(question)

        candidates = repository.hybrid_search(
            connection, question, query_vector, user, include_history
        )
        if not candidates:
            return []

        return self._rerank(question, candidates)

    def _rerank(self, question: str, candidates: list[Hit]) -> list[Hit]:
        numbered = "\n".join(
            f"[{number}] {' '.join(hit.content.split())}"
            for number, hit in enumerate(candidates, start=1)
        )

        result = self.reranker.invoke(
            config.prompt("rerank").format(question=question, candidates=numbered)
        )

        scored = []
        for rating in result.ratings:
            if 1 <= rating.index <= len(candidates):
                hit = candidates[rating.index - 1]
                hit.score = rating.score
                scored.append(hit)

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in scored[: config.TOP_K] if hit.score >= config.MIN_RERANK_SCORE]

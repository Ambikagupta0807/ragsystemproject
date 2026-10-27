# Enterprise RAG System

A production-style Retrieval-Augmented Generation (RAG) system built to answer questions
from internal company documents — grounded, cited, and access-controlled.

## What it does

- Ingests Markdown documents, deduplicating by content hash (not filename), so the same
  policy uploaded under a different name is recognized instead of re-indexed
- Splits documents using a two-level strategy: Markdown headings first, then recursive
  character splitting for any section still too large — so a section is never cut in half
- Embeds chunks with OpenAI embeddings and stores them in PostgreSQL with the pgvector
  extension
- Retrieves relevant chunks using similarity search, with support for MMR (diversity),
  score thresholds, metadata filtering, and an LLM-based reranker
- Generates answers that include word-for-word citations, which are verified with
  deterministic string matching (not a second LLM call) to catch hallucinated quotes
- Implements Corrective RAG: a router decides if retrieval is even needed, a grader
  checks if retrieved chunks are sufficient, and if not, the query is rewritten and
  retried before falling back to an honest "I don't know"

## Why this project

Most RAG tutorials stop at "search + generate." This project focuses on the parts that
matter in production: catching hallucinations, handling vocabulary mismatches, measuring
which chunking strategy actually performs better, and failing gracefully instead of
guessing.

## Tech stack

Python · LangChain · OpenAI API · PostgreSQL + pgvector · FastAPI · Docker · Pydantic

## Key features

- Content-hash based deduplication (no duplicate indexing)
- 5 chunking strategies benchmarked against a labeled test set
- Citation grounding verification (zero extra LLM cost)
- Query rewriting + multi-query retrieval for vocabulary mismatch
- LLM-based reranking for higher precision
- Corrective retrieval loop with automatic query correction

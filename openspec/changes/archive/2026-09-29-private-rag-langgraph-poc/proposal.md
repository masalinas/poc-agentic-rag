# Proposal

## Why

The goal of this Proof of Concept (PoC) is to build a private, agentic RAG backend in Python using LangGraph. It provides a conversational interface capable of performing hybrid search (dense semantic embeddings + sparse lexical full-text search) over uploaded user documents, while maintaining temporal conversational context across chat turns using a lightweight knowledge graph (Graphiti).

## What Changes

- **Python LangGraph Backend**: Core execution engine built using `StateGraph` and checkpointed with `PostgresSaver`.
- **Unified PostgreSQL Data Layer**: Single database instance providing SQL persistent chat checkpoints, `pgvector` dense vector indexing, and `tsvector` sparse lexical full-text search.
- **Hybrid Retrieval System**: Combines vector cosine similarity and full-text keyword search merged via Reciprocal Rank Fusion (RRF).
- **Temporal Conversation Graph**: Integrated with Graphiti to extract entities, relationships, and temporal events from conversation turns to provide time-aware conversational memory.
- **Document Processing Ingest Pipeline**: File upload endpoint, text chunking, and embedding generation.

## Capabilities

### New Capabilities
- `private-rag-agent`: Core RAG agent capability providing hybrid retrieval, temporal graph memory integration, document ingestion, and PostgreSQL state persistence.

### Modified Capabilities
*(None - Greenfield project)*

## Impact

- **New Dependencies**: `langgraph`, `langchain`, `psycopg` / `asyncpg`, `pgvector`, `graphiti-core`, `fastapi`, `uvicorn`, `pydantic`.
- **Infrastructure**: Requires PostgreSQL with `pgvector` extension enabled.

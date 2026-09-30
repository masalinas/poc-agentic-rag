# Design

## Context

See `proposal.md` for motivation. This design specifies the architecture and technical decisions for implementing a private, agentic RAG system in Python using LangGraph, a unified PostgreSQL storage layer (`pgvector` + `tsvector` + `PostgresSaver`), and Graphiti for temporal conversational memory.

## Goals / Non-Goals

**Goals:**
- Architect a clean, modular Python backend using FastAPI and LangGraph.
- Implement a single PostgreSQL storage backend for checkpointer state, dense embeddings (`pgvector`), and sparse full-text search (`tsvector`).
- Combine dense and sparse search using Reciprocal Rank Fusion (RRF).
- Integrate Graphiti to track entities, facts, and temporal context across conversation turns.

**Non-Goals:**
- Building a complex frontend UI (FastAPI Swagger/OpenAPI and raw JSON API endpoints are sufficient for the PoC).
- Multi-node Kubernetes deployment or distributed vector databases (single Dockerized PostgreSQL instance is used for the PoC).

## Decisions

### 1. Unified PostgreSQL Data Store
- **Decision**: Use PostgreSQL as the single database system for vector storage (`pgvector`), lexical search (`tsvector`), and LangGraph checkpointer persistence (`PostgresSaver`).
- **Rationale**: Reduces operational complexity for the PoC by avoiding multiple database servers (e.g. Qdrant + Elasticsearch + Redis).
- **Alternatives Considered**: Dedicated vector DB (Qdrant/Milvus) + BM25 engine. Rejected for initial PoC simplicity.

### 2. Temporal Knowledge Graph with Graphiti
- **Decision**: Use Graphiti as the temporal graph memory layer.
- **Rationale**: Graphiti manages entity extraction, relationships, and temporal valid-from/valid-to boundaries out-of-the-box, providing a lightweight graph memory without requiring a full Neo4j deployment.
- **Alternatives Considered**: Raw NetworkX graph or Neo4j database. Rejected to minimize boilerplate and infrastructure footprint.

### 3. LangGraph State & Node Architecture
- **Decision**: Define an `AgentState` containing `messages`, `user_query`, `documents`, `graph_context`, and `final_response`.
- **Node Flow**:
  1. `query_optimizer_node`: Formulates optimized queries for PostgreSQL hybrid search and Graphiti.
  2. `hybrid_retriever_node`: Executes parallel vector & FTS queries in PostgreSQL and applies RRF ranking.
  3. `temporal_graph_node`: Queries Graphiti for active entity relationships.
  4. `synthesizer_node`: Generates answer using LLM with document chunks and temporal context.
  5. `memory_update_node`: Extracts new facts from the turn and updates Graphiti.
- **Checkpointer**: Use `PostgresSaver` to automatically serialize and persist state transitions per `thread_id`.

## Risks / Trade-offs

- **[Risk] RRF Parameter Tuning** → Default RRF constant $k=60$ works well for general retrieval, but weights between dense vector and FTS can be adjusted dynamically if domain terminology dominates.
- **[Risk] Graph Extraction Latency** → Extracting graph entities on every turn adds an LLM call. *Mitigation*: Run graph updates asynchronously or lazily after generating the user response.

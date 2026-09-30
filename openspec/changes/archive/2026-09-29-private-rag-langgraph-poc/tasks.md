# Tasks

## 1. Project Setup and Infrastructure

- [x] 1.1 Create Python project structure (`app/`, `requirements.txt`, `config.py`) and verify module imports work.
- [x] 1.2 Configure PostgreSQL schema with `pgvector` extension, `document_chunks` table (embedding + tsvector indices), and verify database connectivity.

## 2. Document Ingestion and Hybrid Retrieval

- [x] 2.1 Implement document chunking and PostgreSQL ingestion pipeline in `app/ingest/file_processor.py` and verify chunk insertion with vector embeddings.
- [x] 2.2 Implement hybrid retriever in `app/db/retriever.py` combining dense vector search and `tsvector` FTS with Reciprocal Rank Fusion (RRF) scoring.

## 3. Temporal Graph Memory

- [x] 3.1 Integrate Graphiti wrapper in `app/memory/temporal_graph.py` for extracting entities/events and retrieving active temporal context.

## 4. LangGraph Agent Implementation

- [x] 4.1 Define `AgentState` schema in `app/agent/state.py` and implement agent nodes in `app/agent/nodes.py`.
- [x] 4.2 Assemble `StateGraph` in `app/agent/graph.py` with `PostgresSaver` checkpointer and verify graph compilation.

## 5. FastAPI Endpoints & Integration Testing

- [x] 5.1 Build FastAPI endpoints (`/ingest` and `/chat`) in `app/main.py`.
- [x] 5.2 Test full end-to-end conversation turn with document retrieval, temporal graph memory lookup, and PostgreSQL state checkpointing.

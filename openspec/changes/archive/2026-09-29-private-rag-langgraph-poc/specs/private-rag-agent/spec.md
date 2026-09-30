# Spec Delta: Private RAG Agent

## Purpose

Provides a private agentic RAG capability in Python using LangGraph, PostgreSQL (pgvector + tsvector), and Graphiti for temporal conversational memory.

## ADDED Requirements

### Requirement: Document Ingestion and Chunking
The system SHALL ingest text and document files, segment them into text chunks, generate vector embeddings, and build full-text search indices in PostgreSQL.

#### Scenario: Ingesting a document file
- **WHEN** a user uploads a document file to the ingestion endpoint
- **THEN** the system splits the file into chunks, generates vector embeddings for each chunk, stores them in PostgreSQL with `pgvector` and `tsvector` indices, and returns an ingestion confirmation with chunk counts.

### Requirement: Hybrid Retrieval with Reciprocal Rank Fusion
The system SHALL execute dense semantic vector search and sparse full-text search against PostgreSQL, merging rank scores using Reciprocal Rank Fusion (RRF).

#### Scenario: Hybrid query execution
- **WHEN** a user prompt is processed by the hybrid retriever node
- **THEN** the system queries top-k vector matches and top-k FTS matches from PostgreSQL, computes RRF fused scores, and returns the top-N merged document chunks.

### Requirement: Temporal Conversational Knowledge Graph
The system SHALL maintain a temporal entity graph using Graphiti, extracting facts from user interactions and retrieving relevant entity context for conversational turns.

#### Scenario: Querying temporal context
- **WHEN** a user prompt references a past entity or event
- **THEN** the system queries Graphiti for matching entity states valid for the query timeframe and includes the graph context in the agent prompt.

### Requirement: LangGraph Agent and SQL State Checkpointing
The system SHALL execute the agent workflow using a LangGraph `StateGraph` and persist thread states across turns to PostgreSQL using `PostgresSaver`.

#### Scenario: Conversational turn persistence
- **WHEN** an interaction completes in a given thread
- **THEN** the system saves the full agent state and conversation history checkpoint in PostgreSQL and allows subsequent turns in the same thread to resume seamlessly.

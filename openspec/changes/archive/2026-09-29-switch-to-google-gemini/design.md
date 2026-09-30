# Design

## Context

See `proposal.md` for motivation. This design specifies how to refactor `app/config.py`, `app/ingest/file_processor.py`, and `app/agent/nodes.py` to use Google Gemini models (`gemini-3.6-flash` and `models/text-embedding-004`).

## Goals / Non-Goals

**Goals:**
- Add `langchain-google-genai` and `google-genai` dependencies.
- Refactor `app/config.py` to add `GOOGLE_API_KEY` / `GEMINI_API_KEY`, `MODEL_PROVIDER` ("google"), `LLM_MODEL` ("gemini-3.6-flash"), and `EMBEDDING_MODEL` ("models/text-embedding-004").
- Update `app/agent/nodes.py` to instantiate `ChatGoogleGenerativeAI` when `MODEL_PROVIDER == "google"`.
- Update `app/ingest/file_processor.py` to instantiate `GoogleGenerativeAIEmbeddings` when `MODEL_PROVIDER == "google"`.

**Non-Goals:**
- Removing OpenAI support completely (the configuration can fall back or switch via environment variables).

## Decisions

### 1. LangChain Google GenAI Integration
- **Decision**: Use `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` from `langchain-google-genai`.
- **Rationale**: Provides native compatibility with existing LangChain/LangGraph prompts, chains, and document processing pipeline.

### 2. Dual Key Environment variable support
- **Decision**: Support both `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
- **Rationale**: `GOOGLE_API_KEY` is the standard variable used by `langchain-google-genai` and `google-genai`, while `GEMINI_API_KEY` is often provided by users.

## Risks / Trade-offs

- **[Risk] Missing API Key during offline testing** → Maintain deterministic mock embedding fallback when no API key is set.

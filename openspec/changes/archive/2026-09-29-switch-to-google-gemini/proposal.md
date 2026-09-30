# Proposal

## Why

The user has an API Key for Google Gemini models and wants to use `gemini-3.6-flash` as the LLM model and `models/text-embedding-004` as the embedding model provider in place of OpenAI.

## What Changes

- **Configuration Refactoring**: Update `app/config.py` to include `GOOGLE_API_KEY`, `MODEL_PROVIDER` ("google"), `LLM_MODEL` ("gemini-3.6-flash"), and `EMBEDDING_MODEL` ("models/text-embedding-004").
- **Dependencies**: Add `langchain-google-genai` and `google-genai` to `requirements.txt`.
- **Node Synthesizer Update**: Update `app/agent/nodes.py` to invoke `ChatGoogleGenerativeAI` from `langchain_google_genai`.
- **File Processor Embeddings Update**: Update `app/ingest/file_processor.py` to invoke `GoogleGenerativeAIEmbeddings` from `langchain_google_genai`.

## Capabilities

### New Capabilities
*(None)*

### Modified Capabilities
- `private-rag-agent`: Updates default LLM and embedding provider configuration requirements to support Google Gemini (`gemini-3.6-flash`).

## Impact

- **Dependencies**: Adds `langchain-google-genai` and `google-genai`.
- **Environment**: Requires `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.

# Tasks

## 1. Dependencies and Configuration

- [x] 1.1 Add `langchain-google-genai` and `google-genai` to `requirements.txt` and install packages in virtual environment.
- [x] 1.2 Refactor `app/config.py` to support `GOOGLE_API_KEY`, `MODEL_PROVIDER` ("google"), `LLM_MODEL` ("gemini-3.6-flash"), and `EMBEDDING_MODEL` ("models/text-embedding-004").

## 2. Model Provider Refactoring & Integration Testing

- [x] 2.1 Update `app/ingest/file_processor.py` to generate embeddings using `GoogleGenerativeAIEmbeddings` when provider is Google.
- [x] 2.2 Update `app/agent/nodes.py` to generate responses using `ChatGoogleGenerativeAI` (`gemini-3.6-flash`) when provider is Google.
- [x] 2.3 Run integration test suite `tests/test_e2e.py` and verify Google Gemini backend execution.

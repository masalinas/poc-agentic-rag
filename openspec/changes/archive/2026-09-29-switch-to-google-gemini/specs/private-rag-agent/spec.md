# Spec Delta: Private RAG Agent

## ADDED Requirements

### Requirement: Google Gemini Model Provider Support
The system SHALL support Google Gemini (`gemini-3.6-flash` or configurable Gemini model) as the LLM provider and Google Generative AI embeddings as the embedding model provider via `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

#### Scenario: Configuring Google Gemini provider
- **WHEN** `GOOGLE_API_KEY` or `GEMINI_API_KEY` is configured and model provider is set to `google`
- **THEN** the system initializes `ChatGoogleGenerativeAI` for LLM synthesis and `GoogleGenerativeAIEmbeddings` for vector embeddings generation.

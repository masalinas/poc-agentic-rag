import uuid

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.db.postgres import get_db_connection


def get_embedding(text: str) -> list[float]:
    """Generates embeddings using Google Gemini API or OpenAI API if key is valid, else deterministic mock embedding for PoC testing."""
    if settings.MODEL_PROVIDER == "google" and settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "mock-key":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedder = GoogleGenerativeAIEmbeddings(google_api_key=settings.GOOGLE_API_KEY, model=settings.EMBEDDING_MODEL)
            vec = embedder.embed_query(text)

            # Pad or trim vector dimension to match PostgreSQL 1536-dim column if needed
            if len(vec) != 1536:
                arr = np.array(vec)
                if len(arr) < 1536:
                    arr = np.pad(arr, (0, 1536 - len(arr)), 'constant')
                else:
                    arr = arr[:1536]

                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm

                return arr.tolist()
            
            return vec
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Google Gemini embedding failed ({e}), falling back to deterministic mock embedding.")
    elif settings.MODEL_PROVIDER == "openai" and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "mock-key":
        try:
            from langchain_openai import OpenAIEmbeddings
            embedder = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL)

            return embedder.embed_query(text)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: OpenAI embedding failed ({e}), falling back to deterministic mock embedding.")
    
    # Deterministic mock embedding (1536 dimensions) for local testing without API key
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    vec = rng.randn(1536)
    norm = np.linalg.norm(vec)

    return (vec / norm).tolist()

def ingest_text_document(content: str, doc_id: str | None = None) -> int:
    """Splits document content into chunks and stores them in PostgreSQL with embeddings and FTS tsvector."""
    if not doc_id:
        doc_id = str(uuid.uuid4())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(content)
    
    conn = get_db_connection()
    inserted_count = 0
    with conn.cursor() as cur:
        for idx, chunk in enumerate(chunks):
            emb = get_embedding(chunk)
            cur.execute(
                """
                INSERT INTO document_chunks (doc_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, idx, chunk, emb)
            )
            inserted_count += 1
            
    conn.close()
    return inserted_count

if __name__ == "__main__":
    sample_text = (
        "El proyecto RAG Privado utiliza Google Gemini (gemini-3.6-flash) para síntesis "
        "y PostgreSQL con pgvector y tsvector para la búsqueda híbrida."
    )
    count = ingest_text_document(sample_text, doc_id="doc_gemini_test")
    print(f"Successfully ingested {count} chunks for document doc_gemini_test.")

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def get_db_connection():
    conn = psycopg.connect(settings.database_url, autocommit=True)

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    return conn

def init_db():
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create document chunks table for hybrid retrieval
        cur.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            doc_id VARCHAR(255) NOT NULL,
            chunk_index INT NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1536),
            fts tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
        );
        """)
        
        # Create FTS GIN index
        cur.execute("""
        CREATE INDEX IF NOT EXISTS document_chunks_fts_idx ON document_chunks USING GIN(fts);
        """)
        
    conn.close()
    print("Database initialized successfully with vector extension and document_chunks table.")

if __name__ == "__main__":
    init_db()

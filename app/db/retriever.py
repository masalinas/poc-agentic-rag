from typing import Any

from app.db.postgres import get_db_connection
from app.ingest.file_processor import get_embedding


def vector_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    query_emb = get_embedding(query)
    conn = get_db_connection()
    results = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_id, chunk_index, content, 1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_emb, query_emb, top_k)
        )

        for row in cur.fetchall():
            results.append({
                "id": str(row[0]),
                "doc_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "vector_score": float(row[4])
            })
    conn.close()

    return results

def fts_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    conn = get_db_connection()
    results = []

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_id, chunk_index, content, ts_rank(fts, websearch_to_tsquery('spanish', %s)) AS score
            FROM document_chunks
            WHERE fts @@ websearch_to_tsquery('spanish', %s)
            ORDER BY score DESC
            LIMIT %s;
            """,
            (query, query, top_k)
        )

        for row in cur.fetchall():
            results.append({
                "id": str(row[0]),
                "doc_id": row[1],
                "chunk_index": row[2],
                "content": row[3],
                "fts_score": float(row[4])
            })
    conn.close()

    return results

def hybrid_search(query: str, top_n: int = 5, k_rrf: int = 60) -> list[dict[str, Any]]:
    vec_results = vector_search(query, top_k=20)
    fts_results = fts_search(query, top_k=20)
    
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}
    
    # Process vector rankings
    for rank, item in enumerate(vec_results, start=1):
        chunk_id = item["id"]
        chunk_map[chunk_id] = item
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + rank))
        
    # Process FTS rankings
    for rank, item in enumerate(fts_results, start=1):
        chunk_id = item["id"]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = item
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (k_rrf + rank))
        
    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    final_results = []
    for cid in sorted_ids[:top_n]:
        res = chunk_map[cid].copy()
        res["rrf_score"] = rrf_scores[cid]
        final_results.append(res)
        
    return final_results

if __name__ == "__main__":
    results = hybrid_search("PostgreSQL Graphiti", top_n=3)

    print(f"Hybrid search returned {len(results)} results:")
    for r in results:
        print(f" - [{r['rrf_score']:.4f}] {r['content'][:80]}...")

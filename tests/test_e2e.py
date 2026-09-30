import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_ingest_and_chat_e2e():
    # 1. Ingest test document
    doc_content = (
        "El sistema de RAG Privado utiliza PostgreSQL con pgvector para búsqueda vectorial "
        "y tsvector para búsqueda léxica. También incluye Graphiti para la memoria temporal."
    )
    ingest_res = client.post("/ingest/text", json={"content": doc_content, "doc_id": "test_doc_e2e"})
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    
    # 2. First chat turn (Thread T1)
    thread_id = "thread_test_e2e"
    chat_res1 = client.post("/chat", json={"message": "¿Qué base de datos utiliza el sistema de RAG Privado?", "thread_id": thread_id})
    assert chat_res1.status_code == 200
    res1_data = chat_res1.json()
    assert res1_data["thread_id"] == thread_id
    assert len(res1_data["documents"]) > 0
    
    # 3. Second chat turn (Thread T1 - verifies temporal graph memory and chat history context)
    chat_res2 = client.post("/chat", json={"message": "¿Y qué componente se usa para la memoria temporal?", "thread_id": thread_id})
    assert chat_res2.status_code == 200
    res2_data = chat_res2.json()
    assert len(res2_data["graph_context"]) > 0 or "Graphiti" in res2_data["response"] or "tsvector" in res2_data["response"]

if __name__ == "__main__":
    test_root_endpoint()
    test_ingest_and_chat_e2e()
    print("All E2E Integration Tests Passed Successfully!")

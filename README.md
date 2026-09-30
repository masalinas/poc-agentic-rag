# Private RAG Deep Agent PoC

Un sistema de **RAG Privado y Agente Conversacional Avanzado** (Deep Agent / Agentic RAG) implementado en Python utilizando **LangGraph**, **PostgreSQL** (`pgvector` + `tsvector`), **Graphiti** para memoria semántico-temporal basada en grafos y **Google Gemini** (`gemini-2.5-flash`).

---

## 🏗️ Arquitectura del Sistema

El proyecto sigue una arquitectura unificada y eficiente donde una única instancia de **PostgreSQL** maneja la búsqueda de vectores densos, la búsqueda léxica por texto completo y la persistencia de estados de conversación.

```
+-----------------------------------------------------------------------------------+
|                                 USUARIO / CLIENTE                                 |
+-----------------------------------------------------------------------------------+
       |                                      ^
       | 1. Request POST /chat                | 7. JSON Response (Answer)
       v                                      |
+-----------------------------------------------------------------------------------+
|                        FASTAPI BACKEND & LANGGRAPH AGENT                          |
|                                                                                   |
|  +--------------------+    +-----------------------+    +-----------------------+ |
|  | 1. Query Optimizer | -> |  2. Hybrid Retrieval  | -> |  3. Temporal Graph    | |
|  |     (Node)         |    | (Vector + Lexical FTS)|    |     Query (Node)      | |
|  +--------------------+    +-----------------------+    +-----------------------+ |
|                                                                     |             |
|  +------------------------------------------------------------------+             |
|  |                                                                                |
|  v                                                                                |
|  +--------------------+    +-----------------------+                              |
|  | 4. Synthesizer     | -> | 5. Update Memory &    |                              |
|  | (Gemini 2.5 Flash) |    |    Postgres Checkpoint|                              |
|  +--------------------+    +-----------------------+                              |
+-----------------------------------------------------------------------------------+
    |                   |                     |                   |
    | (Dense Vector     | (Full-Text          | (State            | (Temporal
    |  Cosine Search)   |  Search tsvector)   |  Checkpoints)     |  Entity Graph)
    v                   v                     v                   v
+---------------------------------------------------+       +-------------------+
|                POSTGRESQL DATABASE                |       | GRAPHITI MEMORY   |
|  - document_chunks (embedding, fts, content)      |       | - Fact extraction |
|  - checkpoints (LangGraph PostgresSaver)          |       | - Entity timeline |
+---------------------------------------------------+       +-------------------+
```

---

## 🔄 Flujo de Trabajo del Agente (`StateGraph`)

El workflow del agente está modelado como un grafo de estado en **LangGraph** (`StateGraph`):

```text
                               +--------------------------+
                               |     1. USER PROMPT       |
                               +--------------------------+
                                            |
                                            v
                               +--------------------------+
                               |   2. Query Optimizer     |
                               +--------------------------+
                                            |
                  +-------------------------+-------------------------+
                  |                                                   |
                  v                                                   v
   +-----------------------------+                     +-----------------------------+
   | 3. hybrid_retriever_node    |                     | 4. temporal_graph_node      |
   |                             |                     |                             |
   | Busca en PostgreSQL:        |                     | Busca en Graphiti:          |
   | - Embeddings Vector Coseno  |                     | - Entidades conversacionales|
   | - Búsqueda Léxica FTS BM25  |                     | - Hechos con validez T      |
   +-----------------------------+                     +-----------------------------+
                  |                                                   |
                  +-------------------------+-------------------------+
                                            |
                                            v
                               +--------------------------+
                               |   5. synthesizer_node    |
                               | (LLM Gemini 2.5 Flash)   |
                               +--------------------------+
                                            |
                                            v
                               +--------------------------+
                               | 6. memory_update_node    |
                               | (Guarda turno en Grafo)  |
                               +--------------------------+
```

### Detalle de los Pasos del Workflow:

1. **`1. USER PROMPT`**: Entrada inicial de la petición del usuario con su mensaje y el `thread_id` conversacional.
2. **`2. Query Optimizer`** (`query_optimizer_node`): Analiza el historial reciente de mensajes y genera una consulta optimizada para la recuperación de contexto.
3. **`3. hybrid_retriever_node`**: Ejecución en paralelo sobre **PostgreSQL**:
   - **Búsqueda Semántica**: Vectores densos con distancia coseno en `pgvector`.
   - **Búsqueda Léxica**: Texto completo nativo en español con `tsvector` e índice GIN (BM25).
   - **Fusión RRF**: Los resultados se combinan mediante **Reciprocal Rank Fusion**:
     $$\text{RRF\_Score}(d) = \frac{1}{60 + \text{rank}_{\text{dense}}(d)} + \frac{1}{60 + \text{rank}_{\text{lexical}}(d)}$$
4. **`4. temporal_graph_node`**: Ejecución en paralelo sobre **Graphiti** para extraer entidades conversacionales activas y hechos con validez temporal $T$.
5. **`5. synthesizer_node`**: El modelo LLM **Google Gemini 2.5 Flash** recibe el prompt enriquecido con los Chunks del RAG Híbrido y el Contexto de Grafo Temporal para generar la respuesta final.
6. **`6. memory_update_node`**: Extrae y guarda los nuevos hechos y entidades del turno actual en la memoria de grafo Graphiti, al tiempo que **`PostgresSaver`** persiste el estado del checkpoint en la base de datos PostgreSQL.

---

## 🛠️ Tech Stack

| Componente | Tecnología / Librería | Descripción |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** + **Uvicorn** | API REST en Python para chat e ingesta de documentos |
| **Orquestación Agente** | **LangGraph** (`StateGraph`) | Grafo de estados para orquestar la lógica del agente |
| **Modelo LLM** | **Google Gemini 2.5 Flash** | Sintetizador de respuestas rápido y preciso |
| **Modelo Embeddings** | **Google Generative AI** (`text-embedding-004`) | Generación de vectores densos para chunks |
| **Vector DB** | **PostgreSQL con `pgvector`** | Búsqueda semántica vectorial (índice HNSW/IVFFlat) |
| **Búsqueda Léxica** | **PostgreSQL `tsvector`** | Búsqueda por palabras clave nativa en español (índice GIN) |
| **SQL Checkpointer** | **LangGraph `PostgresSaver`** | Persistencia de conversaciones y estados en PostgreSQL |
| **Memoria de Grafo** | **Graphiti** (`graphiti-core`) | Memoria semántico-temporal basada en grafos de conocimiento |
| **Entorno & Infra** | **Docker Compose**, **`uv`** | Infraestructura PostgreSQL containerizada y gestión Python |

---

## ⚡ Búsqueda Híbrida e Ingesta Automática

- **Ingesta de Documentos**: Los archivos o textos enviados a `/ingest/text` o `/ingest/file` son segmentados mediante `RecursiveCharacterTextSplitter`.
- **Cálculo de FTS Nátivo**: Al insertar los chunks en la tabla `document_chunks`, PostgreSQL calcula e inserta de forma transparente y automática la columna de búsqueda léxica:
  ```sql
  fts tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
  ```

---

## 💡 Ejemplos de Uso

### 1. Ingesta de Texto (`POST /ingest/text`)

Petición cURL para registrar un documento en el sistema:

```bash
curl -X POST "http://localhost:8000/ingest/text" \
     -H "Content-Type: application/json" \
     -d '{
       "content": "El sistema de RAG Privado utiliza PostgreSQL con pgvector y tsvector para búsqueda híbrida. La memoria temporal conversacional está integrada con Graphiti.",
       "doc_id": "manual_rag_01"
     }'
```

**Respuesta JSON:**
```json
{
  "status": "success",
  "doc_id": "manual_rag_01",
  "chunks_inserted": 1
}
```

---

### 2. Conversación Turno 1 - Consulta de Documentos (`POST /chat`)

El usuario consulta sobre el documento ingerido utilizando un `thread_id` específico:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "¿Qué tecnología se utiliza para la búsqueda híbrida en el sistema?",
       "thread_id": "sesion_demo_01"
     }'
```

**Respuesta JSON:**
```json
{
  "thread_id": "sesion_demo_01",
  "response": "El sistema de RAG Privado utiliza PostgreSQL combinando la extensión pgvector (para la búsqueda semántica vectorial) y tsvector (para la búsqueda léxica de texto completo) procesadas mediante Reciprocal Rank Fusion (RRF).",
  "documents": [
    {
      "id": "c1f7b0a8-34b2-4d29-a1b9-823cd1234567",
      "doc_id": "manual_rag_01",
      "chunk_index": 0,
      "content": "El sistema de RAG Privado utiliza PostgreSQL con pgvector y tsvector para búsqueda híbrida...",
      "rrf_score": 0.0328
    }
  ],
  "graph_context": []
}
```

---

### 3. Conversación Turno 2 - Memoria Semántico-Temporal (`POST /chat`)

En el siguiente turno, el usuario pregunta por un hecho conversacional mencionado previamente. El agente recupera el contexto desde el grafo temporal de Graphiti:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "¿Y qué componente mencionamos que gestiona la memoria temporal?",
       "thread_id": "sesion_demo_01"
     }'
```

**Respuesta JSON:**
```json
{
  "thread_id": "sesion_demo_01",
  "response": "La memoria temporal conversacional del sistema está integrada utilizando Graphiti, que rastrea los hechos y entidades mencionadas a lo largo del tiempo.",
  "documents": [],
  "graph_context": [
    "Fact [1790684311.306]: User='¿Qué tecnología se utiliza para la búsqueda híbrida?' -> Assistant='El sistema utiliza PostgreSQL con pgvector y tsvector...'"
  ]
}
```

---

## 🚀 Guía de Inicio Rápido

### 1. Iniciar PostgreSQL con Docker

```bash
docker compose up -d
```

### 2. Configurar la API Key de Google Gemini

Crea un archivo `.env` en la raíz del proyecto:

```bash
GOOGLE_API_KEY=tu_api_key_de_google_aqui
```

### 3. Iniciar el Servidor Backend

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
```

La documentación de Swagger UI estará disponible en **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## 🧪 Pruebas de Integración

Puedes ejecutar la suite de pruebas end-to-end para verificar la ingesta, la búsqueda híbrida, la memoria temporal y los checkpoints en PostgreSQL:

```bash
PYTHONPATH=. .venv/bin/python tests/test_e2e.py
```

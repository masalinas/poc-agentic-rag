import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agent.graph import build_agent_graph, get_postgres_checkpointer
from app.db.postgres import init_db
from app.ingest.file_processor import ingest_text_document

app = FastAPI(title="Private RAG Deep Agent PoC", version="0.1.0")

# Global compiled agent app
agent_app = None

@app.on_event("startup")
def startup_event():
    global agent_app

    print("Initializing Database...")
    init_db()

    print("Compiling LangGraph Agent with Postgres Checkpointer...")
    checkpointer = get_postgres_checkpointer()
    agent_app = build_agent_graph(checkpointer=checkpointer)

    print("Application Startup Complete!")

class TextIngestRequest(BaseModel):
    content: str
    doc_id: str | None = None

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Private RAG Deep Agent Backend",
        "features": ["LangGraph StateGraph", "PostgreSQL pgvector + FTS Hybrid Search", "Graphiti Temporal Graph Memory", "PostgresSaver Checkpoints"]
    }

@app.post("/ingest/text")
def ingest_text(req: TextIngestRequest):
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
        
    doc_id = req.doc_id or str(uuid.uuid4())
    count = ingest_text_document(req.content, doc_id=doc_id)
    return {"status": "success", "doc_id": doc_id, "chunks_inserted": count}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), doc_id: str | None = Form(None)):  # noqa: B008
    contents = await file.read()
    text = contents.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty or invalid text")
        
    target_id = doc_id or file.filename or str(uuid.uuid4())
    count = ingest_text_document(text, doc_id=target_id)
    return {"status": "success", "doc_id": target_id, "filename": file.filename, "chunks_inserted": count}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    global agent_app

    if not agent_app:
        checkpointer = get_postgres_checkpointer()
        agent_app = build_agent_graph(checkpointer=checkpointer)
        
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    input_state = {
        "messages": [HumanMessage(content=req.message)],
        "user_query": req.message,
        "search_query": req.message,
        "thread_id": thread_id,
        "documents": [],
        "graph_context": [],
        "final_response": ""
    }
    
    result = agent_app.invoke(input_state, config=config)
    
    return {
        "thread_id": thread_id,
        "response": result.get("final_response", ""),
        "documents": result.get("documents", []),
        "graph_context": result.get("graph_context", [])
    }

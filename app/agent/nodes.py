from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState
from app.config import settings
from app.db.retriever import hybrid_search
from app.memory.temporal_graph import temporal_memory


def query_optimizer_node(state: AgentState) -> dict:
    """Extracts or rephrases the last user prompt for document and graph retrieval."""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            last_user_msg = msg.content
            break
            
    if not last_user_msg:
        last_user_msg = state.get("user_query", "")
        
    return {"user_query": last_user_msg, "search_query": last_user_msg}

def hybrid_retriever_node(state: AgentState) -> dict:
    """Executes dense vector + tsvector hybrid search with RRF."""
    query = state.get("search_query", "")
    if not query:
        return {"documents": []}
        
    docs = hybrid_search(query, top_n=5)

    return {"documents": docs}

def temporal_graph_node(state: AgentState) -> dict:
    """Retrieves active temporal facts from Graphiti temporal graph memory."""
    query = state.get("search_query", "")
    thread_id = state.get("thread_id", "default")
    graph_ctx = temporal_memory.query_temporal_context(query, thread_id=thread_id)

    return {"graph_context": graph_ctx}

def synthesizer_node(state: AgentState) -> dict:
    """Generates LLM response based on user query, hybrid documents, and temporal graph context using Gemini or OpenAI."""
    user_query = state.get("user_query", "")
    documents = state.get("documents", [])
    graph_context = state.get("graph_context", [])
    
    # Format retrieved document context
    doc_text = "\n".join([f"- Chunk ({d.get('rrf_score', 0):.4f}): {d.get('content', '')}" for d in documents])
    if not doc_text:
        doc_text = "(Ningún documento relevante encontrado)"
        
    # Format temporal graph context
    graph_text = "\n".join([f"- {g}" for g in graph_context])
    if not graph_text:
        graph_text = "(Sin contexto temporal previo)"
        
    prompt = (
        f"Eres un asistente de RAG Privado accionado por el modelo {settings.LLM_MODEL} con acceso a búsqueda híbrida y memoria temporal.\n\n"
        f"PREGUNTA DEL USUARIO: {user_query}\n\n"
        f"CONTEXTO DOCUMENTAL (RAG HÍBRIDO):\n{doc_text}\n\n"
        f"CONTEXTO TEMPORAL DE CONVERSACIÓN (GRAFO):\n{graph_text}\n\n"
        f"Por favor responde de forma precisa basándote en los datos disponibles."
    )
    
    answer = None
    if settings.MODEL_PROVIDER == "google" and settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "mock-key":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.2
            )
            res = llm.invoke(prompt)
            answer = res.content
        except Exception as e:  # noqa: BLE001
            print(f"Warning: ChatGoogleGenerativeAI failed ({e}), falling back to structured output.")
    elif settings.MODEL_PROVIDER == "openai" and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "mock-key":
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model=settings.LLM_MODEL, temperature=0.2)
            res = llm.invoke(prompt)
            answer = res.content
        except Exception as e:  # noqa: BLE001
            print(f"Warning: ChatOpenAI failed ({e}), falling back to structured output.")
            
    if not answer:
        answer = (
            f"Respuesta del Agente RAG Privado ({settings.MODEL_PROVIDER} - {settings.LLM_MODEL}):\n\n"
            f" Contexto de Documentos (Búsqueda Híbrida Vector + FTS):\n{doc_text}\n\n"
            f" Contexto Temporal (Memoria de Grafo Graphiti):\n{graph_text}"
        )
        
    return {
        "final_response": answer,
        "messages": [AIMessage(content=answer)]
    }

def memory_update_node(state: AgentState) -> dict:
    """Records the completed conversational turn in the temporal graph memory."""
    user_query = state.get("user_query", "")
    final_response = state.get("final_response", "")
    thread_id = state.get("thread_id", "default")
    
    if user_query and final_response:
        temporal_memory.add_turn_facts(user_query, final_response, thread_id=thread_id)
        
    return {}

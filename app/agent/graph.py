import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import ConnectionPool

from app.agent.nodes import (
    hybrid_retriever_node,
    memory_update_node,
    query_optimizer_node,
    synthesizer_node,
    temporal_graph_node,
)
from app.agent.state import AgentState
from app.config import settings


def build_agent_graph(checkpointer=None):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_optimizer", query_optimizer_node)
    workflow.add_node("hybrid_retriever", hybrid_retriever_node)
    workflow.add_node("temporal_graph", temporal_graph_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("memory_update", memory_update_node)
    
    # Define edges
    workflow.add_edge(START, "query_optimizer")
    workflow.add_edge("query_optimizer", "hybrid_retriever")
    workflow.add_edge("query_optimizer", "temporal_graph")
    workflow.add_edge("hybrid_retriever", "synthesizer")
    workflow.add_edge("temporal_graph", "synthesizer")
    workflow.add_edge("synthesizer", "memory_update")
    workflow.add_edge("memory_update", END)
    
    return workflow.compile(checkpointer=checkpointer)

def get_postgres_checkpointer():
    """Initializes and returns a PostgresSaver checkpointer."""
    try:
        # Run setup with an autocommit connection
        with psycopg.connect(settings.database_url, autocommit=True) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            
        # Create connection pool for runtime checkpointer
        pool = ConnectionPool(settings.database_url, kwargs={"autocommit": True}, min_size=1, max_size=5)
        return PostgresSaver(pool)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: PostgresSaver setup failed ({e}), compiling graph without checkpointer.")
        return None

if __name__ == "__main__":
    checkpointer = get_postgres_checkpointer()
    agent_app = build_agent_graph(checkpointer=checkpointer)

    print("LangGraph Agent graph compiled successfully!")

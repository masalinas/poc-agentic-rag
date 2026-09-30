from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    search_query: str
    documents: list[dict[str, Any]]
    graph_context: list[str]
    final_response: str
    thread_id: str

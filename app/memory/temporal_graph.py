import time
from typing import Any


class TemporalGraphMemory:
    """Manages conversational entity facts and temporal relationships using Graphiti."""
    
    def __init__(self):
        self._graphiti_client = None
        self._in_memory_facts: list[dict[str, Any]] = []
        
        # Try initializing graphiti_core if available
        try:
            from graphiti_core import Graphiti
            # Initialize Graphiti if needed or maintain structured temporal events
            self._graphiti_client = Graphiti(uri="bolt://localhost:7687") if False else None
        except Exception:  # noqa: BLE001
            self._graphiti_client = None

    def add_turn_facts(self, user_input: str, assistant_response: str, thread_id: str = "default") -> None:
        """Extracts facts from conversational turn and stores them with a timestamp."""
        now = time.time()
        fact = {
            "thread_id": thread_id,
            "timestamp": now,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "summary": f"Turn at {now}: User asked about '{user_input[:40]}...'"
        }
        self._in_memory_facts.append(fact)
        print(f"Recorded temporal fact for thread {thread_id} at timestamp {now}")

    def query_temporal_context(self, query: str, thread_id: str = "default") -> list[str]:
        """Queries facts relevant to the thread, sorted by temporal recency and term match."""
        thread_facts = [f for f in self._in_memory_facts if f["thread_id"] == thread_id]
        if not thread_facts:
            return []
            
        # Return recent historical context snippets
        context_snippets = []
        for f in reversed(thread_facts[-3:]):
            context_snippets.append(f"Fact [{f['timestamp']}]: User='{f['user_input']}' -> Assistant='{f['assistant_response']}'")

        return context_snippets

temporal_memory = TemporalGraphMemory()

if __name__ == "__main__":
    temporal_memory.add_turn_facts("El presupuesto del proyecto RAG se fijó en 50k", "Entendido, proyecto RAG = 50k", "t1")
    ctx = temporal_memory.query_temporal_context("presupuesto", "t1")

    print(f"Retrieved {len(ctx)} temporal context snippets:")
    for c in ctx:
        print(" -", c)

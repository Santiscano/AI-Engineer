from typing import Annotated, TypedDict

from langgraph.graph import add_messages

class RagAgentState(TypedDict):
    """Estados del agente RAG."""
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: list[str]
    response: str
    needs_retrieval: bool
    source: list[dict]
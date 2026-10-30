
from functools import partial
from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from devTalles_python_AI_aplicada.src.graphs.nodes import decide_retrieval_path, node_analyze, node_generate, node_retrieve
from devTalles_python_AI_aplicada.src.graphs.states import RagAgentState

def build_rag_agent(vectorstore: Chroma):
    """Construir y compilar RAG agéntico"""
    builder = StateGraph(RagAgentState)

    node_retrieve_with_vs = partial(node_retrieve, vectorstore=vectorstore)

    builder.add_node("analyze", node_analyze)
    builder.add_node("retrieve", node_retrieve_with_vs)
    builder.add_node("generate", node_generate)

    builder.add_edge(START, "analyze")

    builder.add_conditional_edges(
        "analyze",
        decide_retrieval_path,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )

    builder.add_edge("retrieve", "generate")

    builder.add_edge("generate", END)

    return builder.compile()
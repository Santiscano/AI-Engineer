
import json
from typing import Literal

from langchain.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma

from devTalles_python_AI_aplicada.src.config import settings
from devTalles_python_AI_aplicada.src.core.llm import get_llm
from devTalles_python_AI_aplicada.src.graphs.prompts import get_context_section, get_prompt_template_node_analyzer, get_prompt_template_node_generator
from devTalles_python_AI_aplicada.src.graphs.states import RagAgentState


def node_analyze(state: RagAgentState) -> dict:
    """Nodo que decide si la pregunta necesita buscar documentos"""
    llm = get_llm(temperature=0.1)

    recent_context = ""
    if state.get("messages") and len(state["messages"]) >= 2:
        last_two = state["messages"][-2:]
        recent_context = "\n".join([
            f"{'Usuario' if message.type == 'human' else 'AI'}: {message.content}"
            for message in last_two
            if hasattr(message, 'content') and message.content
        ])

    prompt = get_prompt_template_node_analyzer(recent_context, state)

    result = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = result.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()

        decision = json.loads(content.strip())
        needs_retrieval = decision.get("needs_retrieval", True)
        reason = decision.get("reason", "")
    
    except json.JSONDecodeError:
        needs_retrieval = True
        reason = "Error de parseo, buscando por defecto"

    print(f"[analyze] needs_retrieval={needs_retrieval} | {reason}")
    return {"needs_retrieval": needs_retrieval}



def node_retrieve(state: RagAgentState, vectorstore: Chroma) -> dict:
    """Busca los chunks más relevantes en ChromaDB"""
    print(f" [retrieve] Buscando: '{state['question'][:60]}...'")

    docs = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS}
    ).invoke(state["question"])

    retrieved_texts = [doc.page_content for doc in docs]

    source = [
        {
            "file": doc.metadata.get("file_name", "desconocida"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in docs
    ]
    print(f" [retrieve] {len(docs)} chunks encontrados")

    for src in source:
        print(f" -> {src['file']} (página {src['page']})")

    return {"retrieved_docs": retrieved_texts, "source": source}



def node_generate(state: RagAgentState) -> dict:
    """Genera la respuesta"""
    llm = get_llm(temperature=settings.LOW_TEMPERATURE)

    if state.get("retrieved_docs"):
        docs_text = "\n\n --- \n\n".join(state["retrieved_docs"])
        context_section = get_context_section(docs_text)
    else:
        context_section = "No se encontraron documentos relevantes. Responde con conocimiento general"

    history_text = ""
    if state.get("messages"):
        previous_msgs = state["messages"][:-1]
        recent_msgs = previous_msgs[-6:] if len(previous_msgs) > 6 else previous_msgs
        if recent_msgs:
            history_text = "\n".join([
                f"{'Usuario' if message.type == 'human' else 'Asistente'}: {message.content[:200]}"
                for message in recent_msgs
                if hasattr(message, 'content') and message.content
            ])

    prompt = get_prompt_template_node_generator(context_section, history_text, state)
    result = llm.invoke([HumanMessage(content=prompt)])
    used_docs = "Con documentos" if state.get("retrieved_docs") else "Sin documentos"
    print(f" [generate] {used_docs} ({len(result.content)} chars)")

    return {"response": result.content, "message": [AIMessage(content=result.content)]}



def decide_retrieval_path(state: RagAgentState) -> Literal["retrieve", "generate"]:
    """Función de decisión"""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "generate"
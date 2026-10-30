
from curses.ascii import isdigit
from pathlib import Path

import uuid

from langchain.messages import HumanMessage

from devTalles_python_AI_aplicada.src.graphs.rag_agent import build_rag_agent
from devTalles_python_AI_aplicada.src.graphs.states import RagAgentState

from devTalles_python_AI_aplicada.src.memory.base import BaseMemoryBackend
from devTalles_python_AI_aplicada.src.memory.postgresql_memory import PostgreSQLMemoryBackend
from devTalles_python_AI_aplicada.src.memory.sqlite_memory import SQLiteMemoryBackend
from devTalles_python_AI_aplicada.src.core.document_loader import load_directory, split_documents
from devTalles_python_AI_aplicada.src.core.embeddings import get_or_create_vectorstore

DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "knwoledge_base"
CHROMA_PATH = "./data/chromadb_knwoledge"


def setup_vectorstore():
    """Carga o inicializa el vector store"""
    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME,
        persist_path=CHROMA_PATH
    )

    count = vectorstore._collection.count()

    if count > 0:
        print(f" Base de conocimiento: {count} chunks indexados")
        return vectorstore

    print("Base de conocimiento vacía. Indexando documentos...")
    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        print(f"\n Agrega archivos .txt o .pdf en {DOCUMENTS_DIR}")
        return None

    chunks = split_documents(docs)
    vectorstore.add_documents(chunks)

    print(f"{len(chunks)} chunks indexados")
    return vectorstore


def setup_memory_backend() -> BaseMemoryBackend:
    """Intentar conectar a PostgreSQL o en su defecto usar SQLite"""
    try:
        backend = PostgreSQLMemoryBackend()
        backend.list_sessions()
        print("Memoria: PostgreSQL")
        return backend
    except Exception as e:
        print(f"❌ PostgreSQL no disponible {e}")
        print(" Usando SQLite como fallback de desarrollo")
        return SQLiteMemoryBackend()


def select_session(backend: BaseMemoryBackend) -> str:
    """Seleccionar sesión o crear una nueva"""
    exists_sessions = backend.list_sessions()

    print("\n")
    print("=" * 55)
    print("Gestión de sesiones")
    print("=" * 55)

    if exists_sessions:
        print(f"\nConversaciones guardadas ({len(exists_sessions)}): ")
        for index, session_id in enumerate(exists_sessions, 1):
            messages = backend.get_history(session_id).messages
            last_message = ""
            if messages:
                last_message = f" - último: '{messages[-1].content[:40]}...'"
            print(f" {index}. {session_id}{last_message}")

        print("\nOpciones:")
        print("  n -> Nueva conversación")
        print("  1,2,3... -> Retomar conversación existente")
        print("  ID -> Escribir un session_id específico")

        choise = input("\nElige: ").strip().lower()

        if choise == "n":
            session_id = str(uuid.uuid4())
            print(f"\nNueva sesión: {session_id}")
            return session_id

        if choise.isdigit():
            idx = int(choise) - 1
            if 0 <= idx < len(exists_sessions):
                session_id = exists_sessions[idx]
                messages = backend.get_history(session_id).messages
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
                return session_id
            else:
                print("Número inválido. Creando nueva sesión")

        if choise and choise != "n":
            session_id = choise
            messages = backend.get_history(session_id).messages
            if messages:
                print(f"\n Retomando sesión: {session_id}")
                print(f" {len(messages)} mensajes previos")
            else:
                print(f"\n Nueva sesión con ID: {session_id}")
            return session_id

        session_id = str(uuid.uuid4())
        print(f"\nNueva sesión: {session_id}")
        return session_id

    else:
        session_id = str(uuid.uuid4())
        print(f"\nPrimera sesión: {session_id}")
        return session_id


def load_history(backend: BaseMemoryBackend, session_id: str) -> list:
    """Carga mensajes previos"""
    history = backend.get_history(session_id)
    return list(history.messages)


def save_messages(
        backend: BaseMemoryBackend,
        session_id: str,
        human_message:
        str, ai_message: str) -> None:
    """Persiste el turno de conversación en la base de datos"""
    history = backend.get_history(session_id)
    history.add_user_message(human_message)
    history.add_ai_message(ai_message)


def run_chat(
    agent,
    backend: BaseMemoryBackend,
    session_id: str
) -> None:

    history = load_history(backend, session_id)

    print(f"\n{'=' * 55}")
    print("Asistente de Conocimiento Empresarial")
    print(f"Sesión: {session_id}")
    print(f"{'=' * 55}")
    print("Comandos: 'sesion' | 'historial' | 'limpiar' | 'salir'")
    print("-" * 55)
    print()

    while True:
        try:
            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                messages = backend.get_history(session_id).messages
                print(f"\nSesión guardada: {session_id}")
                print(
                    f"{len(messages)} mensajes en {'PostgreSQL' if isinstance(backend, PostgreSQLMemoryBackend) else 'SQLite'}")
                break

            if user_input.lower() == "sesion":
                print(f"\n ID de la sesión actual: {session_id}")
                messages = backend.get_history(session_id).messages
                print(f"Mensajes en esta sesión: {len(messages)}\n")
                continue

            if user_input.lower() == "historial":
                messages = backend.get_history(session_id).messages

                if not messages:
                    print("[Historial vacío]\n")
                    continue

                print("\núltimos mensajes de la sesión: ")
                for messages in messages[-6:]:
                    rol = "Tú" if messages.type == "human" else "IA"
                    print(f"{rol}: {messages.content[:90]}...")
                print()
                continue

            if user_input.lower() == "limpiar":
                backend.clear_history(session_id)
                history = []
                print("Historial de esta sesión borrada. \n")
                continue

            history = load_history(backend, session_id)

            initial_state: RagAgentState = {
                "messages": history + [HumanMessage(content=user_input)],
                "question": user_input,
                "retrieved_docs": [],
                "response": "",
                "needs_retrieval": True,
                "sources": [],
            }

            print()

            final_state = agent.invoke(initial_state)

            response = final_state["response"]

            user_retrieval = bool(final_state.get("retrieved_docs"))

            if user_retrieval:
                print(f"IA [busco en documentos]: {response}")
            else:
                print(f"IA [respondió directo]: {response}")

            sources = final_state.get("sources", [])

            if sources:
                print("\n Fuentes consultadas")
                shown = set()
                for source in sources:
                    key = f"{source['file']}_p{source['page']}"
                    if key not in shown:
                        print(f"{source['file']} (pág. {source['page']})")
                        shown.add(key)

            print()

            save_messages(backend, session_id, user_input, response)

        except KeyboardInterrupt:
            print("\nHasta luego\n")
            print(f"ID de sesión: {session_id}")
            break

        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> None:
    print("="*50)
    print("Asistente de conocimientos empresarial")
    print("="*50)

    vectorstore = setup_vectorstore()
    if vectorstore is None:
        return

    backend = setup_memory_backend()

    agent = build_rag_agent(vectorstore)
    print("Grafo LangGraph compilado")

    session_id = select_session(backend)

    run_chat(agent, backend, session_id)


if __name__ == "__main__":
    main()
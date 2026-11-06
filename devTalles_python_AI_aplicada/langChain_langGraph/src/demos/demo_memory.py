
from langchain_core.runnables import RunnableWithMessageHistory

from ..memory.postgresql_memory import PostgreSQLMemoryBackend
from ..memory.sqlite_memory import SQLiteMemoryBackend
from ..memory.base import BaseMemoryBackend
from ..chains.base import build_assistant_chain


def build_chatbot(backend: BaseMemoryBackend) -> RunnableWithMessageHistory:
    """Construyendo chatbot
    Args: 
        backend: Cualquie implementación BaseMemory Backend
    Returns:
        Chatbot: listo para invocar con session_id
    """
    chain = build_assistant_chain()

    return RunnableWithMessageHistory(
        chain,
        backend.get_history,
        input_messages_key="input",
        history_messages_key="history"
    )


def run_chat_session(
    chatbot: RunnableWithMessageHistory,
    backend: BaseMemoryBackend,
    session_id: str
) -> None:
    """Sesión de chat interactiva"""
    print(f"\nSesión activa: {session_id}")

    messages = backend.get_history(session_id).messages

    if messages:
        print(
            f"✅ Retomando la conversación ({len(messages)}) mensajes previos")
    else:
        print("Nueva conversación")

    print("Comandos: 'historial' | 'limpiar' | 'sesiones' | 'salir' \n")

    while True:
        try:
            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                print("¡Hasta luego! 👋 Historial guardado")
                break

            if user_input.lower() == "historial":
                messages = backend.get_history(session_id).messages

                if not messages:
                    print(" [Historial vacío]\n")
                    continue
                print(f"\nÚltimos mensajes de '{session_id}'")

                for message in messages[-6:]:
                    rol = "Tú" if message.type == "human" else "IA"
                    print(f" {rol}: {message.content[:70]}...")
                print()
                continue

            if user_input.lower() == "limpiar":
                backend.clear_history(session_id)
                print("HISTORIAL BORRADO.\n")
                continue

            if user_input.lower() == "sesiones":
                sessions = backend.list_sessions()
                print(f"\n Sesiones disponibles: {sessions}\n")
                continue

            response = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}} # con el config, se pasa el session_id al backend para que pueda recuperar el historial de la sesión correspondiente
            )

            print(f"\nIA: {response}\n")
        except KeyboardInterrupt:
            print("¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    print("="*60)
    print("Memoria persistente con SQLite y PostgreSQL")
    print("="*60)

    print("\n¿Qué backend memory usar?")
    print(" 1.SQLite (archivo local)")
    print(" 2.PostgreSQL (requiere Docker ejecutandose)")

    choice = input("Elige(1/2): ").strip()

    if choice == "2":
        try:
            backend = PostgreSQLMemoryBackend()
            print("✅ Conectado a PostgreSQL")
        except ValueError as e:
            print(f"❌ {e}")
            print("Usando SQLite como respaldo...")
            backend = SQLiteMemoryBackend()
    else:
        backend = SQLiteMemoryBackend()
        print("✅ Conectado a SQLite")

    chatbot = build_chatbot(backend)
    run_chat_session(chatbot, backend, session_id="user_demo_001")
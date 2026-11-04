from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def count_approximate_tokens(messages: list) -> int:
    """
    Estimación simple de tokens en el historial.
    """
    total_text = " ".join(
        message["content"] for message in messages
        if isinstance(message["content"], str)
    )
    return len(total_text)


def chat_without_management(history: list, new_message: str) -> str:
    """Chatbot naive: acumula historial infinitamente."""
    history.append({"role": "user", "content": new_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    estimated_tokens = count_approximate_tokens(history)
    limit = 128_000  # gpt-4o-mini
    percentage = (estimated_tokens / limit) * 100

    print(f"  [Tokens estimados: {estimated_tokens:,} / {limit:,} "
          f"({percentage:.1f}% de la ventana)]")

    return answer


history = [
    {
        "role": "system",
        "content": "Eres un asistente de soporte técnico para una empresa de software."
    }
]

conversations = [
    "Mi nombre es Carlos y trabajo en el área de ventas.",
    "Tengo un problema con el módulo de reportes, no carga.",
    "El error dice: 'ConnectionTimeout after 30s'.",
    "¿Cuál era mi número de ticket?",
]

print("=" * 60)
print("DEMOSTRACIÓN: Chatbot sin gestión de contexto")
print("=" * 60)

for message in conversations:
    print(f"\nUsuario: {message}")
    answer = chat_without_management(history, message)
    print(f"IA: {answer[:200]}...")  # Truncamos para legibilidad

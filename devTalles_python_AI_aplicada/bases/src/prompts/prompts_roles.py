"""
Roles en prompts
"""
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def show_roles():
    """Muestra el comportamiento de cada uno de los roles"""
    # Rol User
    print("="*50)
    print("Rol: User. (Sin rol system)")
    print("="*50)

    response_1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "¿Cuánto es 2+2"}
        ]
    )

    print(f"Respuesta: {response_1.choices[0].message.content}\n")

    # Rol System
    print("="*50)
    print("Rol: System")
    print("="*50)

    response_2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Eres un matemático gruñon que contesta
                preguntas simples con desdén pero presición absoluta.
                Siempre incluyes un comentario sobre lo básico que es la
                pregunta
                """
            },
            {"role": "user", "content": "¿Cuánto es 2+2"}
        ]
    )

    print(f"Respuesta: {response_2.choices[0].message.content}\n")

    # Rol Assistant

    print("="*50)
    print("Rol: Assistant")
    print("="*50)

    response_3 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "Eres un clasificador de sentimientos."
             "Respondes SOLO con : POSITIVO, NEGATIVO, o NEUTRO."},
            {"role": "user", "content": "Me encanta el helado"},
            {"role": "assistant", "content": "POSITIVO"},
            {"role": "user", "content": "El clima es templado."},
            {"role": "assistant", "content": "NEUTRO"},
            {"role": "user", "content": "Odio los lunes"},
            {"role": "assistant", "content": "NEGATIVO"},
            {"role": "user", "content": "Hace frío"},
        ]
    )

    print(f"Sentimiento: {response_3.choices[0].message.content}\n")


if __name__ == "__main__":
    show_roles()

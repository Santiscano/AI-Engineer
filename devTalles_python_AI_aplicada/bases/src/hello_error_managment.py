"""Hello error managment"""
from http import client

from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()


def call_ai(question: str) -> str:
    """Función para llamar a la API de OpenAI"""
    client = OpenAI()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=500,
            temperature=1.2  # 0=determinista, 1=creativo, 2=caótico
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print("APIKEY inválido. Revisa tu archivo .env")
        raise SystemExit(1)

    except RateLimitError:
        print("Límite de velocidad alcanzado. Espera un momento.")
        raise

    except APIConnectionError:
        print("Sin conexión a la API. Revisa tu internet.")

    except Exception as e:
        print(f"Error inesperado: {type(e).__name__}: {e}") # accedemos al nombre del tipo de error con type(e).__name__ y al mensaje de error
        raise


if __name__ == "__main__":
    response = call_ai("Dame un título para un video de Python")
    print(f"AI: {response}")
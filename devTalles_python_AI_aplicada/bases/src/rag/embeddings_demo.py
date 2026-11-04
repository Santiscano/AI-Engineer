from http import client
import os
import math
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def get_embedding(text: str) -> list[float]:
    """Convierte texto a vector 1536 dimensiones."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calcula que tan similares son dos vectores"""

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a ** 2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def demonstrate_semantic_similarity():
    """Muestra similitud de significado"""

    # Pregunta
    base_phrase = input("User: ")

    # Documentos
    candidates = [
        "Para reiniciar el servidor ejecuta: sudo systemctl restart nginx",
        "Puedes reboot el proceso con el comando service stop/start",
        "The server restart procedure is documented in section 4.2",
        "La pizza margarita lleva tomate, mozzarella y albahaca",
        "Los gatos domésticos duermen un promedio de 16 horas al día",
        "Para apagar el servidor usa: sudo shutdown -h now",
    ]

    print("Calculando embeddings...")

    base_embedding = get_embedding(base_phrase)
    results = []

    for phrase in candidates:
        candidate_embedding = get_embedding(phrase)
        similarity = cosine_similarity(base_embedding, candidate_embedding)
        results.append((similarity, phrase))

    results.sort(reverse=True)

    print(f"\nPregunta: {base_phrase}")
    print("\nResultados ordenados por similitud: ")
    print("="*40)

    for similarity, phrase in results:
        bar = "█" * int(similarity*30)
        relevance = "RELEVANTE" if similarity > 0.5 else "IRRELEVANTE"

        print(f"{similarity:.3f} {bar}")
        print(f"{relevance}: {phrase[:60]}...")


if __name__ == "__main__":
    print("="*60)
    print("Embeddings: Busqueda por vector")
    print("="*60)

    demonstrate_semantic_similarity()

    # embedding_vector = get_embedding("Hombre")
    # embedding_vector_b = get_embedding("Mujer")

    # similarity = cosine_similarity(embedding_vector, embedding_vector_b)

    # print(similarity)

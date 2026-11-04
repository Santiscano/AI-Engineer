import os
from unittest import result
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
import uuid


load_dotenv()

openai_client = OpenAI()


class RAGPipeline:
    """Pipeline RAG completo"""

    def __init__(self, collection_name: str, db_path: str = "./data/chromadb"):
        """Inicializando el pipeline con ChromaDB"""
        self.chroma_client = chromadb.PersistentClient(path=db_path)

        # embeddingFunction es la función que convierte texto en vectores para búsqueda semántica
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"

        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        print(f"RAG Pipeline iniciado, Colección: {collection_name}"
              f"Documentos: {self.collection.count()}")

    def index_texts(
        self,
        texts: list[str],
        metadatas: list[dict] = None
    ) -> None:
        """Agrega textos a la base de conocimientos"""
        if not texts:
            return

        ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in texts]

        if metadatas is None:
            metadatas = [{"fuente": "manual"} for _ in texts]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        print(f"{len(texts)} fragmentos indexados"
              f"Total en DB: {self.collection.count()}")

    def index_chunks(
        self,
        long_text: str,
        chunk_size: int = 500,
        overlap: int = 50,
        base_metadata: dict = None
    ) -> int:
        """Divide un texto largo en chunks"""

        chunk_chars_size = chunk_size * 4

        chunks = []
        start = 0

        while start < len(long_text):
            end = start + chunk_chars_size
            chunk = long_text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start = end - (overlap*4)

        metadatas = []

        for i, _ in enumerate(chunks):
            meta = (base_metadata or {}).copy()
            meta["chunk_numero"] = i
            meta["chunk_total"] = len(chunks)
            metadatas.append(meta)
        self.index_texts(chunks, metadatas)

        return len(chunks)

    def retrieve_context(
        self,
        question: str,
        n_fragments: int = 3
    ) -> list[dict]:
        """Busca los fragmentos más relevantes para una pregunta"""
        results = self.collection.query(
            query_texts=[question],
            n_results=min(n_fragments, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        fragments = []

        for i in range(len(results["documents"][0])):
            similarity = round(1 - (results["distances"][0][i]), 3)

            if similarity > 0.3:
                fragments.append({
                    "texto": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similitud": similarity
                })
        return fragments

    def answer(
        self,
        question: str,
        n_fragments: int = 3,
        verbose: bool = False
    ) -> dict:
        """Pipeline completo RAG"""

        # 1. Recupear contexto relevante
        fragments = self.retrieve_context(question, n_fragments)

        if not fragments:
            return {
                "respuesta": "No encontré información relevante en la base de conocimiento.",
                "fragmentos_usados": [],
                "tiene_contexto": False
            }

        if verbose:
            print(f"\nFragmentos recuperados para: {question}")

            for fragment in fragments:
                print(
                    f"[{fragment['similitud']}]"
                    f"{fragment['texto'][:80]}..."
                )

        # 2. Construir el contexto para LLM
        context_text = "\n\n--\n\n".join([
            f"[Fuente: {fragment['metadata'].get('fuente', 'desconocida')}]\n"
            f"{fragment['texto']}"
            for fragment in fragments
        ])

        # 3. Generar respuesta con el LLM usando el contexto

        system_prompt = """Eres un asistente experto que responde preguntas
        basándote ÚNICAMENTE en el contexto proporcionado.
        Reglas:
        - Si la respuesta está en el contexto, respóndela directamente y con precisión.
        - Si el contexto no contiene suficiente información, dilo honestamente.
        - Cita la fuente cuando sea relevante.
        - No inventes información que no esté en el contexto.
        - Responde en el mismo idioma de la pregunta."""

        user_prompt = f"""Contexto disponible:
        {context_text}
        Pregunta: {question}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        return {
            "respuesta": response.choices[0].message.content,
            "fragmentos_usados": fragments,
            "tokens_usados": response.usage.total_tokens,
            "tiene_contexto": True
        }


if __name__ == "__main__":

    DOCUMENTS = [
        {
            "texto": "Python fue creado por Guido van Rossum y lanzado en 1991. "
                     "Es un lenguaje de programación de alto nivel, interpretado y de propósito general.",
            "metadata": {"fuente": "python_history.txt", "tema": "historia"}
        },
        {
            "texto": "Las listas en Python son colecciones ordenadas y mutables. "
                     "Se crean con corchetes: mi_lista = [1, 2, 3]. "
                     "Puedes agregar elementos con .append() y eliminar con .remove().",
            "metadata": {"fuente": "python_basics.txt", "tema": "estructuras_datos"}
        },
        {
            "texto": "Los decoradores en Python son funciones que modifican el comportamiento "
                     "de otras funciones. Se usan con la sintaxis @nombre_decorador. "
                     "Son muy comunes en frameworks como FastAPI y Django.",
            "metadata": {"fuente": "python_advanced.txt", "tema": "avanzado"}
        },
        {
            "texto": "Para manejar errores en Python se usa try/except. "
                     "Ejemplo: try: resultado = 10/0 except ZeroDivisionError: print('División por cero'). "
                     "También existe finally para código que siempre se ejecuta.",
            "metadata": {"fuente": "python_basics.txt", "tema": "manejo_errores"}
        },
        {
            "texto": "Los virtual environments (entornos virtuales) en Python aislan "
                     "las dependencias de cada proyecto. Se crean con: python -m venv .venv "
                     "y se activan con: source .venv/bin/activate en Linux/Mac.",
            "metadata": {"fuente": "python_setup.txt", "tema": "configuracion"}
        },
    ]

    print("="*60)
    print("RAG PIPELINE")
    print("="*60)

    # Incializamos pipeline
    rag = RAGPipeline("python_knowledge_base")

    if rag.collection.count() == 0:
        print("\nIndexando base de conocimientos...\n")
        texts = [doc["texto"] for doc in DOCUMENTS]
        metas = [doc["metadata"] for doc in DOCUMENTS]
        rag.index_texts(texts, metas)

    # Preguntas
    questions = [
        "¿Quién creo Python?",
        "¿Cómo manejo excepciones en Python?",
        "¿Para que sirven los decoradores?",
        "¿Cómo instalo Django?"
    ]

    print("\n")
    print("="*60)
    print("Consultas al sistema RAG")
    print("="*60)

    for question in questions:
        print(f"\nPREGUNTA: {question}")

        result = rag.answer(question, n_fragments=2, verbose=True)

        print(f"RESPUESTA:\n {result['respuesta']}")
        print("**Metricas**")
        print(f" Tokens usados: {result.get('tokens_usados', 'N/A')}")
        print(f" Fragmentos: {len(result['fragmentos_usados'])}")
        print(f" Tiene contexto: {result['tiene_contexto']}")



from pathlib import Path

from ..chains.rag import build_rag_chain
from ..core.embeddings import get_or_create_vectorstore
from ..core.document_loader import load_directory, split_documents


DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "demo_rag_real"
CHROMA_PATH = "./data/chromadb_rag_demo"


def index_documents() -> tuple:
    """Carga archivos reales desde disco y divide en chunks"""
    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        return None, 0  # (vectorstore, num_chunks)

    chunks = split_documents(docs)

    if not chunks:
        print("❌ No se generaron chunks. Los archivos podrían estar vacíos.")
        return None, 0

    # crear o cargar una colleccion en ChromaDB
    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME,
        persist_path=CHROMA_PATH
    )

    current_count = vectorstore._collection.count()

    if current_count > 0:
        print(f"\n. Ya hay {current_count} chunks indexados")
        answer = input("¿Reindexar desde cero? (s/N): ").strip().lower()

        if answer == "s":
            vectorstore._client.delete_collection(COLLECTION_NAME)
            vectorstore = get_or_create_vectorstore(
                collection_name=COLLECTION_NAME,
                persist_path=CHROMA_PATH
            )

            vectorstore.add_documents(chunks)
            print(f"Reindexado: {len(chunks)} chunks en ChromaDB")

        else:
            print(f"Usando índice existente ({current_count} chunks)")

    else:
        vectorstore.add_documents(chunks)
        print(f"{len(chunks)} chunks indexados en ChromaDB")

    return vectorstore, vectorstore._collection.count()


def show_used_sources(docs: list) -> None:
    """Muestra archivos y fragmentos que usó el sistema"""
    if not docs:
        return

    print("\nFuentes consultadas: ")

    viewed_sources = {}

    for doc in docs:
        name = doc.metadata.get("file_name", "desconocida")
        page = doc.metadata.get("page", None)
        start = doc.metadata.get("start_index", None)

        if name not in viewed_sources:
            viewed_sources[name] = []

        info = ""

        if page is not None:
            info = f"pág. {page+1}"
        elif start is not None:
            info = f"pos. {start}"

        if info:
            viewed_sources[name].append(info)

    for file, locations in viewed_sources.items():
        if locations:
            print(f" {file} ({', '.join(locations)})")
        else:
            print(f" {file}")


def show_welcome(num_chunks: int) -> None:
    """Muestra el mensaje de bienvenida con estado del sistema."""
    print("\n" + "=" * 60)
    print("🔍 RAG Interactivo — Chat con tus documentos")
    print("=" * 60)
    print(f"  Carpeta de documentos: {DOCUMENTS_DIR}")
    print(f"  Chunks en el índice:   {num_chunks}")
    print()
    print("  Comandos:")
    print("    'archivos'  → ver archivos indexados")
    print("    'reindexar' → recargar archivos del disco")
    print("    'chunks'    → ver cantidad de chunks")
    print("    'salir'     → terminar")
    print("-" * 60)
    print()


def main() -> None:
    """Main"""
    print("=" * 60)
    print("Iniciando RAG con archivos reales...")
    print("=" * 60)

    vectorstore, num_chunks = index_documents()

    if vectorstore is None:
        print("\n Pasos para empezar")
        print(f" 1. Crea la carpera: {DOCUMENTS_DIR}")
        print(" 2. Agrega archivos .txt o .pdf")
        print(" 3. Vuelve a ejecutar este script")
        return

    rag_chain, retriever = build_rag_chain(vectorstore)

    show_welcome(num_chunks)

    while True:
        try:
            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                print("¡Hasta luego!")
                break

            if user_input.lower() == "archivos":
                files = list(DOCUMENTS_DIR.glob("*.txt")) + list(DOCUMENTS_DIR.glob("*.pdf")) + \
                    list(DOCUMENTS_DIR.glob("*.TXT")) + \
                    list(DOCUMENTS_DIR.glob("*.PDF"))
                files = list(set(files))

                if not files:
                    print(f" NO hay archivos en {DOCUMENTS_DIR}\n")

                else:
                    print(f"\n Archivos en {DOCUMENTS_DIR}")
                    for file in sorted(files):
                        size_kb = file.stat().st_size / 1024
                        print(f" {file.name} ({size_kb:.1f}) KB")
                    print()
                continue

            if user_input.lower() == "chunks":
                count = vectorstore._collection.count()
                print(f"\n Chunks en ChromaDB: {count}\n")
                continue

            if user_input.lower() == "reindexar":
                print("\nReindexando...")
                vectorstore, num_chunks = index_documents()
                if vectorstore:
                    rag_chain, retriever = build_rag_chain(vectorstore)
                    print(f"Listo. {num_chunks} chunks disponibles.\n")
                continue

            retrieved_docs = retriever.invoke(user_input)

            if not retrieved_docs:
                print("\n IA: No encontré información relevante en los documentos.\n")
                continue

            print("\n IA: ", end="", flush=True)
            answer = rag_chain.invoke(user_input)
            print(answer)

            show_used_sources(retrieved_docs)
            print()
        except KeyboardInterrupt:
            print("\n¡Hasta luego!\n")
            break
        except Exception as e:
            print(f"\n ❌ Error: {e}\n")


if __name__ == "__main__":
    main()
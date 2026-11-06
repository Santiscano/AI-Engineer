
import os

from langchain_chroma import Chroma

from devTalles_python_AI_aplicada.langChain_langGraph.src.core.llm import get_embeddings
from devTalles_python_AI_aplicada.langChain_langGraph.src.config.settings import settings


def get_or_create_vectorstore(
    collection_name: str,
    persist_path: str = None,
) -> Chroma:
    """Obtener un vector store existente o crear uno nuevo"""
    path = persist_path or settings.CHROMA_PATH # obtener la ruta de persistencia desde la configuración si no se proporciona
    os.makedirs(path, exist_ok=True) # crear la carpeta si no existe

    # instanciar chroma con la función de embeddings y la ruta de persistencia
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(), # obtener la función de embeddings desde la configuración
        persist_directory=path,
    )
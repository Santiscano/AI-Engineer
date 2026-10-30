
import os

from langchain_chroma import Chroma

from devTalles_python_AI_aplicada.src.core.llm import get_embeddings
from devTalles_python_AI_aplicada.src.config.settings import settings


def get_or_create_vectorstore(
    collection_name: str,
    persist_path: str = None,
) -> Chroma:
    """Obtener un vector store existente o crear uno nuevo"""
    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=path,
    )
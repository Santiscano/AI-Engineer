"""Configuración de la aplicación"""

import os
from dotenv import load_dotenv

load_dotenv()

class LangChainSettings:
    """Configuración de LangChain"""
    # Modelos
    CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Parámetros LLM
    DEFAULT_TEMPERATURE: float = 0.7
    LOW_TEMPERATURE: float = 0.1
    MAX_RETRIES: int = 3

    # Rutas de persistencia
    SQLITE_DB_PATH: str = "data/chat_history.db"
    CHROMA_PATH: str = "./data/langchain_chroma"

    # RAG
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 3

    # Costos aproximados
    COST_INPUT_PER_MILLION: float = 0.15
    COST_OUTPUT_PER_MILLION: float = 0.60

    @classmethod
    def validate(cls):
        """Valida variables críticas de configuración"""

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "La variable de entorno OPENAI_API_KEY no está configurada.",
                "Por favor, configúrela antes de ejecutar la aplicación."
            )

        cls.validate() # esta linea asegura que la validación se ejecute al importar el módulo

settings = LangChainSettings()
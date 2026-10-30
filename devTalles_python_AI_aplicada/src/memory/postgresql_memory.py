

import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, text

from devTalles_python_AI_aplicada.src.memory.base import BaseMemoryBackend

class PostgreSQLMemoryBackend(BaseMemoryBackend):
    """Backend de memoria usando PostgreSQL"""

    def __init__(self, database_url: str = None):
        self._database_url = database_url or os.getenv("DATABASE_URL")

        if not self._database_url:
            raise ValueError(
                "DATABASE_URL no está configurada."
                "Agrega una url a tu .env"
                "postgresql://user:password@host:port/db"
            )

        if not self._database_url.startswith("postgresql"):
            raise ValueError(
                "DATABASE_URL debe empezar con 'postgresql://' "
            )

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sesión"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self._database_url
        )

    def clear_history(self, session_id: str) -> None:
        """Elimina el historial de una sesión"""
        history = self.get_history(session_id)
        history.clear()

    def list_sessions(self) -> list[str]:
        """Lista todos los session_ids disponibles"""
        engine = create_engine(self._database_url)
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT DISTINCT session_id FROM message_store")
                )

                return [row[0] for row in result]
        except ImportError:
            return []
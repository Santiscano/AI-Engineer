"""Centralized environment-based configuration."""

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


# dataclass es un decorador que se utiliza para crear clases de datos en python. 
# Proporciona una forma concisa de definir clases que contienen principalmente atributos de datos, 
# evitando la necesidad de escribir métodos como __init__ y __repr__ manualmente.
# frozen=True hace que la instancia de la clase sea inmutable, 
# lo que significa que sus atributos no se pueden modificar después de la creación del objeto.
@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    llm_temperature: float
    api_key: str
    log_level: str
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float


def _read_str(name: str, default: str) -> str:
    """Se encarga de leer una variable de entorno y devolver su valor como una cadena de texto."""
    return os.getenv(name, default).strip()


def _read_float(name: str, default: str) -> float:
    """Se encarga de leer una variable de entorno y devolver su valor como un número flotante."""
    raw_value = _read_str(name, default)
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid value for {name!r}: {raw_value!r}. Expected a float."
        ) from exc


# lru_cache es un decorador que se utiliza para almacenar en caché los resultados de una función,
# de modo que si se llama a la función con los mismos argumentos nuevamente, se devolverá el resultado almacenado en caché.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """"""
    llm_provider = _read_str("LLM_PROVIDER", "gemini").lower()
    llm_model = _read_str(
        "LLM_MODEL",
        _read_str("GEMINI_MODEL", "gemini-3-flash-preview"),
    )
    llm_temperature = _read_float("LLM_TEMPERATURE", "0.2")
    api_key = _read_str("API_KEY", "")
    log_level = _read_str("LOG_LEVEL", "INFO").upper()
    input_cost_per_1m_tokens = _read_float("INPUT_COST_PER_1M_TOKENS", "0.0")
    output_cost_per_1m_tokens = _read_float("OUTPUT_COST_PER_1M_TOKENS", "0.0")

    if llm_provider == "gemini" and not api_key:
        raise ValueError(
            "Missing API_KEY. Set it in your environment or .env file."
        )

    return Settings(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        api_key=api_key,
        log_level=log_level,
        input_cost_per_1m_tokens=input_cost_per_1m_tokens,
        output_cost_per_1m_tokens=output_cost_per_1m_tokens
    )
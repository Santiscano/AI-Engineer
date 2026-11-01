"""
News extractor
"""
import json
import os
from dotenv import load_dotenv
from src.helpers.ai_client import call_ai
from src.services.news_service import NewsService

load_dotenv()


def run_news_extractor():
    """News extractor exec"""
    print("Extractor de noticias")
    print("="*40)

#     news = """
# Apple anunció hoy que Tim Cook presentará el nuevo iPhone 17 Pro
# el próximo 15 de septiembre de 2025 en Cupertino, California.
# El dispositivo costará desde $1,199 USD y contará con chip A19.
# """

    news_service = NewsService(os.getenv("NEWS_APIKEY"))
    news = news_service.get_latest_tech_news()

    response_extractor = call_ai([
        {
            "role": "system",
            "content": """Eres un extactor de información de noticias. Si la noticia está en otro idioma traduce al Español.
            Extrae entidades y devuelve solo JSON válido con esta estructura:
            {
                "company": string,
                "person": string,
                "product": string,
                "top_news": string,
                "keywords": string,
                "datetime": string(formato ISO: YYYY-MM-DD),
                "place": string,
                "price": number or null
            }
            """
        },
        {
            "role": "user",
            "content": news
        }
    ],
        0.1,
        "json_object"
    )

    extract_entities = json.loads(response_extractor)
    for key, value in extract_entities.items():
        print(f"{key}: {value}")

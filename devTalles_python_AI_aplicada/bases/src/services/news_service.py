import requests


class NewsService:
    """News service class"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/top-headlines"

    def get_latest_tech_news(self) -> str:
        """Trae las últimas noticias"""
        params = {
            "q": "apple",
            "from": "2026-04-01",
            "sortBy": "popularity",
            "apiKey": self.api_key
        }

        response = requests.get(self.base_url, params=params)
        data = response.json()

        if not data.get("articles"):
            return "No hay noticias disponibles"

        article = data["articles"][0]

        return f"""
        {article.get('title', '')}.
        {article.get('description', '')}.
        {article.get('content', '')}.
    """

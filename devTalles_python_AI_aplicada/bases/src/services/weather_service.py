
import requests


class WeatherService:

    def get_coordinates(self, city: str):
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city,
            "count": 1,
            "language": "es"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data.get("results"):
            return None, None

        result = data["results"][0]
        return result["latitude"], result["longitude"]

    def get_current_weather_by_city(self, city: str):
        lat, lon = self.get_coordinates(city)

        if lat is None:
            return {"error": "Ciudad no encontrada"}

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "auto"
        }

        response = requests.get(url, params=params)

        data = response.json()

        return data.get("current_weather", {})

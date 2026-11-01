
import json
from src.helpers.ai_client import call_ai_tools
from src.services.weather_service import WeatherService

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtiene el clima actual de una ciudad. Usar cuando el usuario pregunta por el tiempo o clima",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "El nombre de la ciudad, ej: 'Madrid' o 'Mexico City'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unidad de temperatura"
                    }
                },
                "required": ["city"]  # Campos obligatorios
            }
        }
    }
]


def get_weather(city: str, unit: str = "celsius") -> dict:
    """Obtener clima"""

    weather_service = WeatherService()

    weather = weather_service.get_current_weather_by_city(city)
    print(weather["temperature"])

    temp = weather["temperature"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return {
        "city": city,
        "temperature": f"{temp}˚{'C' if unit == 'celsius' else 'F'}",
        "wind_speed": f"{weather.get('windspeed', 0)} km/h"
    }


def execute_tool(name: str, arguments: dict) -> str:
    """Mapea el nombre de la función real"""

    available_functions = {
        "get_weather": get_weather,
    }

    if name not in available_functions:
        return json.dumps({"error": f"Función '{name}' no encontrada"})

    result = available_functions[name](**arguments)

    return json.dumps(result, ensure_ascii=False)  # México : "M\u00e9xico"


def run_chat_with_tools(user_message: str) -> str:

    messages = [
        {"role": "system", "content": "Eres un asistente útil con acceso a herramientas."},
        {"role": "user", "content": user_message}
    ]

    print(f"\nUsuario: {user_message}")

    message_ia = call_ai_tools(messages, 0.1, "text", TOOLS, "auto")

    if message_ia.tool_calls:
        print(f"IA decide usar herramientas")
        messages.append(message_ia)

        for tool_call in message_ia.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            print(f" {function_name}({arguments})")

            result = execute_tool(function_name, arguments)

            print(f"RESULTADO: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        final_response = call_ai_tools(messages, 0.1, "text", TOOLS)
    else:
        final_response = message_ia

    print(f"IA: {final_response.content}")
    return final_response.content

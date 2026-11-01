"""Helper para crear y ejecutar el cliente"""
from openai import OpenAI, omit
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def call_ai(messages: list, temperature: float = 0.1, response_format: str = "text") -> str:
    """Función que ejecuta el cliente"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        response_format={"type": response_format}
    )

    return response.choices[0].message.content


def call_ai_tools(messages: list, temperature: float = 0.1, response_format: str = "text", tools: list = omit, tool_choise: str = omit) -> str:
    """Función que ejecuta el cliente con tool"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        response_format={"type": response_format},
        tools=tools,
        tool_choice=tool_choise
    )

    return response.choices[0].message
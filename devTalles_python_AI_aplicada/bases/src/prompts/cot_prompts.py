from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule

from src.helpers.ai_client import call_ai

console = Console()


def run_chain_of_thought():
    """Función de cadena de pensamiento"""
    console.print(Rule("[bold yellow] Chain of Thought"))

    problem = """
    Una empresa tiene 3 servidores. Cada servidor maneja 1,200 request/hora. 
    Tiene picos de 4,500 request/hora los lunes. 
    ¿Cuántos servidores adicionales necesitan para los picos?
    """

    console.print(
        Panel(problem.strip(), title="Problema", border_style="blue"))

    without_cot = call_ai([
        {"role": "user", "content": f"Response solo el número: {problem}"}
    ])

    with_cot = call_ai([
        {"role": "user", "content": f""" 
         {problem}
         
         Piensa paso a paso:
         1. Calcula la capacidad actual.
         2. Calcula el déficit en pico
         3. Determina cuántos servidores adicionales se necesitan
         4. Da la respuesta final
         """}
    ])

    console.print(
        Panel(f"[bold] Sin CoT: [/bold] {without_cot} ", border_style="red"))

    console.print(
        Panel(Markdown(with_cot), title="Con Chain-of-Thought", border_style="green"))

from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from src.helpers.ai_client import call_ai

console = Console()


def run_zero_few_shot():
    console.print(Rule("[bold yellow]Técnica: Zero-Shot y Few_shot"))
    zero_shot = call_ai([
        {
            "role": "user",
            "content": "Clasifica este email como URGENTE O NORMAL: "
            "'El servidor de producción está caído, los clientes no pueden acceder'"
        }
    ])

    few_shot = call_ai([
        {"role": "system", "content": "Clasifica emails. Responde solo con: URGENTE o NORMAL. Sin explicación"},
        {"role": "user", "content": "Tengo una reunión mañana a las 3pm"},
        {"role": "assistant", "content": "NORMAL"},
        {"role": "user", "content": "La base de datos se corrompió en producción"},
        {"role": "assistant", "content": "URGENTE"},
        {"role": "user", "content": "¿Puedes revisar mi PR cuando puedas?"},
        {"role": "assistant", "content": "NORMAL"},
        {"role": "user", "content": "El servidor de producción está caído, los clientes no pueden acceder"},

    ])

    table = Table(title="Comparación")
    table.add_column("Técnica", style="cyan")
    table.add_column("Resultado", style="magenta")

    table.add_row("Zero-shot", zero_shot)
    table.add_row("Few-shot", few_shot)

    console.print(table)

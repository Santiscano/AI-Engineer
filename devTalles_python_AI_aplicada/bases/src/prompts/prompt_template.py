
from rich import prompt
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.rule import Rule
from src.helpers.ai_client import call_ai


console = Console()


def create_code_analysis_prompt(
    code: str,
    language: str,
    detail_level: str = "medium"
) -> str:

    levels = {
        "basic": "Identifica solo bugs críticos",
        "medium": "Identifica bugs críticos, sugiere mejoras de rendimiento y legibilidad",
        "expert": "Análisis completo: bugs, seguridad, rendimiento, patrones de diseño"
    }

    return f""" Analiza el siguiente código {language}.
Nivel de análisis requerido: {levels.get(detail_level, levels["medium"])}
Lenguaje: {language}
Código:
{code}

"""


def run_prompt_templates():
    console.print(Rule("[bold yellow]Propmt Template"))

    example_code = """
    def calcular_promedio(numeros):
        total=0
        for n in numeros:
            total = total+n
        return total/len(numeros)
    """

    syntax = Syntax(example_code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Código a analizar",
                  border_style="cyan"))

    prompt = create_code_analysis_prompt(
        code=example_code,
        language="python",
        detail_level="expert"
    )

    response = call_ai([{"role": "user", "content": prompt}])

    console.print(
        Panel(
            Markdown(response),
            title="Análisis del código",
            border_style="green"
        )
    )

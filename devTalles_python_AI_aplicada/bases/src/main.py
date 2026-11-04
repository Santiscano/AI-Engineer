
from rich.console import Console
from rich.panel import Panel

# from src.prompts.function_calling import run_chat_with_tools
from devTalles_python_AI_aplicada.bases.src.prompts.pydantic_ia import run_cv_project

# from src.prompts.json_mode import run_json_mode
# from src.prompts.news_extractor import run_news_extractor


console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Técnicas de prompting\n"
        )
    )

    # run_json_mode()
    # run_news_extractor()
    # run_chat_with_tools("¿Qué clima hace en Madrid?")
    # run_chat_with_tools("¿Qué clima hace en Mexico City?")
    # run_chat_with_tools("¿Qué clima hace en London?")
    # run_chat_with_tools("¿Qué clima hace en Buenos Aires?")
    # run_chat_with_tools("¿Qué clima hace en Lima?")
    # run_chat_with_tools("¿Qué clima hace en Tampico?")
    run_cv_project()
    console.print("\n[bold green]Ejecución completada\n")


if __name__ == "__main__":
    main()
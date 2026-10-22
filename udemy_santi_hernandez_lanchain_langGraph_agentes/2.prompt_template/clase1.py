from langchain_core.prompts import PromptTemplate

template = """Eres un experto en marketing digital. sugiere una estrategia de marketing para vender {producto}. La estrategia debe incluir los siguientes elementos: 1) Identificación del público objetivo, 2) Canales de marketing a utilizar, 3) Mensaje clave a comunicar, 4) Presupuesto estimado y 5) Cronograma de implementación. Por favor, proporciona una respuesta detallada y estructurada."""

prompt = PromptTemplate(
    template=template,
    input_variables=["producto"]
)

prompt_complete = prompt.format(producto="sesiones de fotografia de bodas, cumpleaños y primera comunion")
print(prompt_complete)
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
import json
from dotenv import load_dotenv
import os

# import variables de entorno
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# configurar modelo
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

# procesador
def preprocess_text(text):
    """Limpiar el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]
preprocessor = RunnableLambda(preprocess_text)


# generar resumen
def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oracion: {text}"
    response = llm.invoke(prompt)
    content = response.content
    print("Response summary: ", content[0]["text"], "\n")
    
    # Si viene como lista, lo unimos a texto plano
    if isinstance(content, list):
        content = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
    return content

summary_branch = RunnableLambda(generate_summary)


# analisis de sentimiento formato JSON
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde UNICAMENTE en formato JSON valido, sin bloques de codigo markdown ni texto adicional.
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificacion breve"}}
    
    Texto: {text}
    """
    
    response = llm.invoke(prompt)
    content = response.content
    print("content analyze sentiment: ", content[0]["text"], "\n")
    
    # 1. Si es una lista, convertimos o unimos los elementos a string
    if isinstance(content, list):
        content = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in content])
    
    content = str(content).strip()
    
    # 2. Limpiar etiquetas de markdown comunes (ej. ```json ... ```) si el modelo las incluye
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en analisis", "raw_content": content}

sentiment_branch = RunnableLambda(analyze_sentiment)


# combinacion resultados
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }
merger = RunnableLambda(merge_results)


parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch
})


chain = preprocessor | parallel_analysis | merger


# Invocar 1 texto de ejemplo
result = chain.invoke("""El ser padre es el regalo mas grande que se puede tener en la vida del hombre, tener un proposito le da sentido a tu existencia,  este proposito nunca es enfocado en el bienestar propio sino en traer seguridad, felicidad y paz a otros, en el caso del padre a sus hijos. \n""")

print(result)

# invocar un lote de textos de ejemplo
batch_texts = [
    "La vida es un viaje lleno de desafíos y aprendizajes. Cada obstáculo nos enseña algo valioso.",
    "El clima hoy es soleado y perfecto para un paseo al aire libre.",
    "La película que vimos anoche fue aburrida y predecible, no la recomendaría."
]
results_batch = chain.batch(batch_texts)
for i, res in enumerate(results_batch):
    print(f"Resultado para texto {i+1}: {res}\n")
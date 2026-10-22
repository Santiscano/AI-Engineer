from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# configurar modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# procesador
def preprocess_text(text):
    """Limpiar el texto eliminando espacos extras y limitando longitud"""
    return text.strip()[:500]

prepocessor = RunnableLambda(preprocess_text)

# generar resumen
def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = f"Resume en una sola oracion: {text}"
    response = llm.invoke(prompt)
    return response.content

# analisis de sentimiento formato JSON
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde UNICAMENTE en formato JSON valido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificacion breve"}}
    
    Texto: {text}
    """
    
    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en analisis"}
    


# combinacion resultdos
def merge_results(data):
    """Combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }
    

def process_one(t):
    resumen = generate_summary(t)
    sentimiento_data = analyze_sentiment(t)
    return merge_results(resumen, sentimiento_data)

process = RunnableLambda(process_one)

chain = prepocessor | process
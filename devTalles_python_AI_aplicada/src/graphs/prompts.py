


from devTalles_python_AI_aplicada.src.graphs.states import RagAgentState


def get_prompt_template_node_analyzer(recent_context: str, state: RagAgentState) -> str:
    prompt = f"""Analiza si esta pregunta necesita buscar en la base de conocimiento empresarial.
    Contexto inmediato (últimos 2 mensajes):
    {recent_context if recent_context else 'Sin contexto previo'}
    Pregunta actual: {state['question']}
    Responde ÚNICAMENTE con este JSON (sin markdown):
    {{"needs_retrieval": true, "reason": "razón breve"}}
    REGLAS ESTRICTAS:
    needs_retrieval = true SIEMPRE que:
    - La pregunta pida información específica (políticas, procedimientos, datos)
    - Mencione documentos, manuales, contratos o información interna
    - Sea una pregunta factual sobre la empresa o sus procesos
    - Haya cualquier duda
    needs_retrieval = false SOLO cuando sea OBVIO:
    - Saludos puros: "hola", "gracias", "adiós"
    - Aclaración de lo que la IA dijo en el mensaje inmediato anterior
    En caso de duda: needs_retrieval = true"""
    return prompt



# node generator
def get_context_section(docs_text: str) -> str:
    prompt = f"""INFORMACION DE LOS DOCUMENTOS EMPRESARIALES: {docs_text}
    INSTRUCCIÓN: Basa tu respuesta principalmente en estos documentos.
    Si la información no está aquí, dilo claramente.
    """
    return prompt

def get_prompt_template_node_generator(context_section: str, history_text: str, state: RagAgentState) -> str:
    prompt = f"""Eres un asistente de conocimiento empresarial experto.
    {context_section}
    HISTORIAL RECIENTE:
    {history_text if history_text else 'Inicio de conversación'}
    PREGUNTA: {state['question']}
    INSTRUCCIONES:
    - Si tienes documentos, úsalos como fuente principal
    - Cita los documentos cuando sea relevante
    - Si algo no está en los documentos, dilo honestamente
    - Usa el historial solo para referencias contextuales
    - Responde en español de forma clara y profesional"""
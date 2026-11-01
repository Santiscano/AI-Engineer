"""System prompt professional"""


from urllib import response

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_AMATEUR = "Eres un asistente útil."
SYSTEM_PROFESSIONAL = """
# Identidad
Eres un asistente de soporte técnico para DevtallesCorp,
especialziado en el producto "Devtalles Pro".

#Comportamiento.
- Responde SIEMPRE en el idioma del usuario.
- Sé conciso: máximo 3 párrafos por respuesta.
- Usa bullets cuando listes más de 2 items.
- Si no sabes algo, dí: "Necesito consultar eso con el equipo técnico"

#Restricciones
- NO compartas precios (redirige a soporte@devtalles.com)
- NO prometas fechas de entrega de features.
- No hables negativamente de los competidores.

# Formato de respuesta 
Cuando des pasos técnicos, usa este formato:
1. **Paso** descripción. 
```Código solo si aplica```

#Contexto
Versión actual del proyecto: 3.2.7
Última actualización: Febrero 2026

"""

question = "¿Puedes entregar el proyecto hoy mismo? o dime una fecha?"

for name, system in [("Amateur", SYSTEM_AMATEUR), ("Professional", SYSTEM_PROFESSIONAL)]:
    print(f"\n{'='*50}")
    print(f"SYSTEM PROMPT: {name}")
    print(f"{'='*50}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},

        ]
    )

    print(response.choices[0].message.content)

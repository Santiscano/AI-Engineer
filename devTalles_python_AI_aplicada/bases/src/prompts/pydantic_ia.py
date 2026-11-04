import json
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

from src.helpers.ai_client import call_ai
from src.services.cvs_service import read_docx

load_dotenv()


class Skill(BaseModel):
    name: str
    level: str  # "basic", "intermediate", "advanced"
    years_experience: float


class CandidateCV(BaseModel):
    """Model representing extracted CV information."""
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    job_title: str
    total_years_experience: float = Field(ge=0, le=50)
    technical_skills: list[Skill]
    languages: list[str]
    candidate_score: float = Field(ge=0, le=10)
    score_reason: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email')
        return v


def extract_cv(cv_text: str) -> CandidateCV:
    schema = CandidateCV.model_json_schema()

    response = call_ai([
        {
            "role": "system",
            "content": f"""Eres un extractor de CV.
Regresa solo el JSON válido siguiende este schema:
{json.dumps(schema, indent=2)}

Evalúa al candidato para un puesto de Desarrollador Backend en Python con un enfoque profesional.

Criterios de evaluación:

Relevancia técnica (alto peso): Solo considera tecnologías directamente relacionadas con backend en Python (APIs, bases de datos, autenticación, arquitectura backend). Ignora tecnologías no relevantes.
Profundidad técnica: Penaliza conocimientos superficiales. Da mayor puntuación a experiencia sólida y comprobable en herramientas clave.
Experiencia práctica real: Prioriza experiencia en proyectos reales, sistemas en producción, escalabilidad, manejo de errores y buenas prácticas.
Buenas prácticas de ingeniería: Evalúa conocimiento en arquitectura limpia, testing, diseño de sistemas, organización del código y mantenibilidad.

Asigna una puntuación del 1 al 10 basada en estos criterios y justifica claramente la calificación.
"""
        },
        {
            "role": "user",
            "content": f"Extract structured data from this CV:\n\n{cv_text}"
        }
    ],
        response_format="json_object",
    )

    data = json.loads(response)
    candidate = CandidateCV(**data)

    return candidate


def process_cv_docx(file_path: str) -> CandidateCV:
    cv_text = read_docx(file_path)
    return extract_cv(cv_text)


def run_cv_project():

    candiate_input = input("Ingresa CV a evaluar: ")
    print("Processing CV...\n")
    try:
        candidate = process_cv_docx(candiate_input)

        print(f"Success!\n")
        print(f"Name: {candidate.full_name}")
        print(f"Title: {candidate.job_title}")
        print(f"Experience: {candidate.total_years_experience} years")
        print(f"Email: {candidate.email}")

        print("\nSkills:")
        for skill in candidate.technical_skills:
            print(
                f" - {skill.name}: {skill.level} ({skill.years_experience} years)")

        print(f"\nLanguages: {', '.join(candidate.languages)}")
        print(f"\nScore: {candidate.candidate_score}/10")
        print(f"Reason: {candidate.score_reason}")

        print("\nJSON:")
        print(candidate.model_dump_json(indent=2))

    except Exception as e:
        print(f"Error: {e}")

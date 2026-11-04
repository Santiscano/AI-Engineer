"""Document read"""
from docx import Document


def read_docx(file_path: str) -> str:
    # Extrae texto del archivo .docx
    doc = Document(file_path)
    text = []

    for paragraph in doc.paragraphs:
        text.append(paragraph.text)

    return "\n".join(text)

"""Chat con PDFs"""
import sys

from rich.panel import Panel
from rich.table import Table
from devTalles_python_AI_aplicada.bases.src.rag.pipeline_rag import RAGPipeline
from pathlib import Path
from pypdf import PdfReader
from rich.console import Console
from dotenv import load_dotenv

load_dotenv() # Cargar variables de entorno desde un archivo .env
console = Console() # herramienta para tener una consola más bonita y con colores, gracias a la librería rich


class PDFProcessor:
    """Leer PDFs y devolver texto"""

    @staticmethod
    def extract_text(pdf_path: Path) -> str: # metodo que recibe la ruta de un pdf y devuelve el texto extraído
        """Extrae el texto de un PDF"""
        try:
            reader = PdfReader(str(pdf_path)) # leer el pdf y crear un objeto PdfReader
            pages_text = []

            for page_number, page in enumerate(reader.pages, 1): # enumerate hace que se tenga un indice de reader.pages y asi poder iterar sobre cada página del pdf
                page_text = page.extract_text() # extraer el texto de la página

                if page_text and page_text.strip(): # si el texto de la página no está vacío, se agrega a la lista pages_text
                    pages_text.append(f"[Página {page_number}]\n{page_text}")

            return "\n\n".join(pages_text) # convierte en texto la lista pages_text transformada en el for anterior

        except Exception as e:
            console.print(f"[red]Error leyendo {pdf_path.name}: {e}[/red]")
            return ""

    @staticmethod
    def get_pdfs(folder: Path) -> list[Path]: # metodo que recibe la ruta de una carpeta y devuelve una lista de rutas de los pdfs que hay en esa carpeta
        """Extraer la lista de archivos de una carpeta"""
        if not folder.exists(): # si la carpeta no existe, se crea, asi evita error
            folder.mkdir(parents=True, exist_ok=True)

        return list(folder.glob("*.pdf")) # devuelve una lista de rutas de los pdfs que hay en la carpeta especificada por folder


class IndexRegistry:
    """Clase para registrar pdf indexados"""

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self.registry: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """De archivo a memoria"""
        if not self.path.exists():
            return

        with open(self.path, "r") as file:
            for line in file.read().splitlines():
                if not line.strip():
                    continue

                parts = line.rsplit(":", 1)  # manual.pdf:204700

                if len(parts) == 2:
                    name, size = parts
                    self.registry[name] = int(size)

    def save(self) -> None:
        """De memoria a archivo"""
        self.path.parent.mkdir(parents=True, exist_ok=True) # se encarga de crear las carpetas intermedias si no existen, para evitar errores al guardar el archivo
        with open(self.path, "w") as file:
            for name, size in self.registry.items():
                file.write(f"{name}:{size}\n")  # manual.pdf:204700

    def is_indexed(self, pdf_path: Path) -> bool: # detectar el cambio de tamaño del pdf, si el tamaño es diferente al registrado, significa que el pdf ha cambiado y debe ser reindexado
        """Comprobar si está indexado"""
        if pdf_path.name not in self.registry: # si el nombre del pdf no está en el registro, significa que no ha sido indexado
            return False

        current_size = pdf_path.stat().st_size # stat().st_size devuelve el tamaño del archivo en bytes.
        return self.registry[pdf_path.name] == current_size # si el tamaño del pdf es igual al registrado, significa que no ha cambiado y por lo tanto está indexado.

    def mark_indexed(self, pdf_path: Path) -> None: # metodo para marcar un pdf como indexado, guardando su tamaño en el registro
        """Marcar como indexado"""
        self.registry[pdf_path.name] = pdf_path.stat().st_size # stat().st_size devuelve el tamaño del archivo en bytes, y se guarda en el registro con el nombre del pdf como clave

    # registry.indexed_names() registry.indexed_names
    @property # convierte el método en una propiedad, para poder acceder a él como un atributo sin necesidad de llamarlo como función
    def indexed_names(self) -> list[str]:
        """Obtener registros"""
        return sorted(self.registry.keys())

    @property
    def count(self) -> int:
        """Contar registros"""
        return len(self.registry)


class ChatWithPDFs:
    """Chat con PDFs"""

    def __init__(self, pdf_folder: str = "data/files/pdfs"):
        self.pdf_folder = Path(pdf_folder)
        self.processor = PDFProcessor()
        self.registry = IndexRegistry(Path("data/pdfs_indexados.txt"))

        self.rag = RAGPipeline(
            collection_name="mis_pdfs",
            db_path="./data/chromadb_pdfs"
        )

    def index_new_pdfs(self) -> int:
        """Indexar nuevos pdfs"""
        pdfs = self.processor.get_pdfs(self.pdf_folder) # obtiene la lista de pdfs en la carpeta especificada por self.pdf_folder

        if not pdfs:
            console.print(
                f"\n[yellow]No hay PDFs en '{self.pdf_folder}'[/yellow]\n"
                f"[dim]Coloca archivos .pdf ahí y escribe 'reindexar'[/dim]"
            )
            return 0

        # 1.Filtrar PDFs que no están indexados
        news_pdfs = [pdf for pdf in pdfs if not self.registry.is_indexed(pdf)]

        if not news_pdfs:
            console.print(
                f"[green]✅ Todos los PDFs ya están indexados"
                f"({len(pdfs)} archivos) [/green]"
            )

            return 0

        console.print(
            f"\n[bold cyan] Indexando {len(news_pdfs)} PDF(s) nuevo(s)...[/bold cyan]")

        indexed_count = 0

        for pdf_path in news_pdfs: # 2.1 iterar sobre cada pdf que no está indexado
            console.print(f"Procesando: [bold]{pdf_path.name}[/bold]", end=" ")
            text = self.processor.extract_text(pdf_path) # 2.1 extrae el texto del pdf

            if not text.strip(): # validar si contiene texto extraible.
                console.print(
                    "[red]❌ Sin texto extraíble[/red]\n"
                    "[dim](puede ser un PDF escaneado - necesitaría OCR)[/dim]"
                )
                continue

            # 2.2 indexa el chunk y guarda el tamaño del pdf en el registro
            num_chunks = self.rag.index_chunks(
                long_text=text,
                chunk_size=400,
                overlap=40,
                base_metadata={
                    "fuente": pdf_path.name,
                    "tipo": "pdf"
                }
            )

            # "La cápital de Francia es Paris" -> "La cápital de Francia", "Francia es Paris"
            console.print(f"[green] ✅ {num_chunks} fragmentos[/green]")
            self.registry.mark_indexed(pdf_path) # marcar el pdf como indexado, guardando su tamaño en el registro
            indexed_count += 1 # contador de pdfs indexados

        self.registry.save()
        return indexed_count

    def show_status(self) -> None:
        """Muestra el estatus de los PDFs"""
        table = Table(title="Estado del Sistema RAG", show_header=True)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")

        table.add_row("PDFs indexados", str(self.registry.count))
        table.add_row("Fragmentos en ChromaDB",
                      str(self.rag.collection.count()))
        table.add_row("Carpeta de PDFs", str(self.pdf_folder))
        table.add_row("Modelo embeddings", "text-embedding-3-small")
        table.add_row("Modelo respuesta", "gpt-4o-mini")

        console.print(table)

        if self.registry.indexed_names:
            console.print("\n[bold]PDFs en la base de conocimiento:[/bold]")
            for name in self.registry.indexed_names:
                console.print(f"  📄 {name}")

    def chat(self) -> None:
        """Loop principal de chat interactivo."""

        console.print(Panel.fit(
            "[bold cyan]Chat con tus PDFs[/bold cyan]\n"
            "[dim]Comandos disponibles: 'estado' | 'reindexar' | 'salir'[/dim]",
            border_style="cyan"
        ))

        if self.rag.collection.count() == 0:
            console.print(
                "\n[yellow]⚠️  La base de conocimiento está vacía.[/yellow]\n"
                f"Coloca PDFs en [bold]{self.pdf_folder}[/bold] "
                "y escribe 'reindexar'\n"
            )

        while True:
            try:
                console.print()
                question = console.input(
                    "[bold green]Tú: [/bold green]").strip()

                if not question:
                    continue

                if question.lower() in ("salir", "exit", "quit"):
                    console.print("[dim]¡Hasta luego! 👋[/dim]")
                    break

                if question.lower() == "estado":
                    self.show_status()
                    continue

                if question.lower() == "reindexar":
                    self.index_new_pdfs()
                    continue

                if self.rag.collection.count() == 0:
                    console.print(
                        "[red]No hay documentos indexados"
                        "Agrega PDFs y escribe 'reindexar'[/red]"
                    )
                    continue

                # RAG
                with console.status("[dim]Buscando en tus documentos...[/dim]"): # el with se uso con el objetivo de que se cierre automáticamente al finalizar
                    result = self.rag.answer(
                        question=question,
                        n_fragments=3,
                        verbose=False
                    )
                # Respuesta
                console.print()
                console.print(Panel(
                    result["respuesta"],
                    title="[bold blue]Asistente[/bold blue]",
                    border_style="blue"
                ))

                if result.get("fragmentos_usados"):
                    console.print("[dim]Fuentes consultadas:[/dim]")
                    seen = set()

                    for fragment in result["fragmentos_usados"]:
                        source = fragment["metadata"].get(
                            "fuente", "desconocida")

                        if source not in seen:
                            console.print(
                                f"[dim] {source}"
                                f"(similitud: {fragment['similitud']})[/dim]"
                            )
                            seen.add(source)

            except KeyboardInterrupt:
                console.print("[dim]¡Hasta luego! 👋[/dim]")
                break

            except Exception as e:
                console.print(f"[red]Error inesperado: {e}[/red]")


def main():
    """Punto de entrada"""
    Path("data/files/pdfs").mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]Iniciando Chat con PDFs...[/bold]\n")

    system = ChatWithPDFs(pdf_folder="data/files/pdfs")
    system.index_new_pdfs()
    system.show_status()
    system.chat()


if __name__ == "__main__":
    main()

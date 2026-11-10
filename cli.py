import typer
from rich.console import Console
from rich.table import Table
from src.engine import SearchEngine
from src.ingest import parse_directory

app = typer.Typer(help="HartsOfflineToolkit: Local Hybrid Knowledge Search")
console = Console()
engine = None

def get_engine():
    global engine
    if engine is None:
        with console.status("[bold cyan]Initializing Engine & Loading Local Weights..."):
            engine = SearchEngine()
    return engine

@app.command()
def index(path: str = typer.Option("./data/raw", help="Directory path to parse and index")):
    eng = get_engine()
    console.print(f"[bold blue]Scanning directory:[/] {path}")
    docs = parse_directory(path)
    if not docs:
        console.print("[yellow]No supported documents found to index.[/]")
        return
    count = eng.index_documents(docs)
    console.print(f"[bold green]Successfully indexed {count} document sections into LanceDB.[/]")

@app.command()
def search(query: str = typer.Argument(..., help="Search query"), limit: int = 3):
    eng = get_engine()
    results = eng.hybrid_search(query, top_n=limit)
    if not results:
        console.print("[yellow]No relevant matches found.[/]")
        return

    table = Table(title=f"Results for: '{query}'", show_lines=True)
    table.add_column("Rank", justify="center", style="cyan", width=6)
    table.add_column("Score (RRF)", justify="center", style="magenta", width=12)
    table.add_column("Document ID", style="green", width=25)
    table.add_column("Content Snippet", style="white")

    for i, res in enumerate(results, 1):
        snippet = res["content"][:280] + "..." if len(res["content"]) > 280 else res["content"]
        table.add_row(str(i), f"{res['score']:.5f}", res["doc_id"], snippet)

    console.print(table)

if __name__ == "__main__":
    app()

import os
import sys
import requests
import argparse
import json
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Initialize rich console for better output formatting
console = Console()

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def ingest_document(self, file_path: str, strategy: str = "paragraph", 
                        chunk_size: int = 1000, overlap: int = 100,
                        api_key: Optional[str] = None, title: Optional[str] = None,
                        author: Optional[str] = None, date: Optional[str] = None) -> Dict[str, Any]:
        """Ingest a document into the RAG system"""
        # Check if file exists
        if not os.path.exists(file_path):
            console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
            sys.exit(1)
        
        # Prepare form data
        files = {"document": open(file_path, "rb")}
        data = {
            "strategy": strategy,
            "chunk_size": str(chunk_size),
            "overlap": str(overlap)
        }
        
        # Add optional fields if provided
        if api_key:
            data["api_key"] = api_key
        if title:
            data["title"] = title
        if author:
            data["author"] = author
        if date:
            data["date"] = date
        
        try:
            response = requests.post(f"{self.base_url}/ingest", files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                console.print(f"Response: {e.response.text}")
            sys.exit(1)
        finally:
            files["document"].close()
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": query_text, "top_k": top_k}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                console.print(f"Response: {e.response.text}")
            sys.exit(1)
    
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the RAG system"""
        try:
            response = requests.get(f"{self.base_url}/documents")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                console.print(f"Response: {e.response.text}")
            sys.exit(1)
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document from the RAG system"""
        try:
            response = requests.delete(f"{self.base_url}/documents/{doc_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                console.print(f"Response: {e.response.text}")
            sys.exit(1)

def display_query_result(result: Dict[str, Any]):
    """Display the query result in a nice format"""
    # Display the answer
    console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
    
    # Display the sources
    if result.get("sources"):
        table = Table(title="Sources")
        table.add_column("ID", style="dim")
        table.add_column("Score", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Text Preview", style="yellow")
        
        for source in result["sources"]:
            text_preview = source["text"][:100] + "..." if len(source["text"]) > 100 else source["text"]
            source_name = source["metadata"].get("source", "Unknown")
            table.add_row(
                source["id"],
                f"{source['score']:.4f}",
                source_name,
                text_preview
            )
        
        console.print(table)

def display_documents(documents: List[Dict[str, Any]]):
    """Display the list of documents in a nice format"""
    if not documents:
        console.print("[yellow]No documents found in the system[/yellow]")
        return
    
    table = Table(title=f"Documents ({len(documents)})")
    table.add_column("ID", style="dim")
    table.add_column("Source", style="green")
    table.add_column("Title", style="cyan")
    table.add_column("Text Preview", style="yellow")
    
    for doc in documents:
        title = doc["metadata"].get("title", "Untitled")
        source = doc["metadata"].get("source", "Unknown")
        table.add_row(
            doc["id"],
            source,
            title,
            doc["text"]
        )
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="RAG System Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document")
    ingest_parser.add_argument("file_path", help="Path to the document file")
    ingest_parser.add_argument("--strategy", default="paragraph", 
                              choices=["semantic", "fixed", "paragraph", "heading", "page"],
                              help="Chunking strategy")
    ingest_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    ingest_parser.add_argument("--overlap", type=int, default=100, help="Overlap size")
    ingest_parser.add_argument("--api-key", help="OpenAI API key (required for semantic chunking)")
    ingest_parser.add_argument("--title", help="Document title")
    ingest_parser.add_argument("--author", help="Document author")
    ingest_parser.add_argument("--date", help="Document date")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query_text", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    
    # List documents command
    list_parser = subparsers.add_parser("list", help="List all documents")
    
    # Delete document command
    delete_parser = subparsers.add_parser("delete", help="Delete a document")
    delete_parser.add_argument("doc_id", help="Document ID to delete")
    
    # Server URL option
    parser.add_argument("--server", default="http://localhost:8000", help="RAG server URL")
    
    args = parser.parse_args()
    
    # Initialize client
    client = RAGClient(base_url=args.server)
    
    if args.command == "ingest":
        result = client.ingest_document(
            file_path=args.file_path,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            api_key=args.api_key,
            title=args.title,
            author=args.author,
            date=args.date
        )
        console.print(f"[bold green]Success:[/bold green] {result['message']}")
        console.print(f"Chunks count: {result['chunks_count']}")
        console.print(f"Strategy: {result['strategy']}")
    
    elif args.command == "query":
        result = client.query(args.query_text, args.top_k)
        display_query_result(result)
    
    elif args.command == "list":
        result = client.list_documents()
        display_documents(result.get("documents", []))
    
    elif args.command == "delete":
        result = client.delete_document(args.doc_id)
        console.print(f"[bold green]Success:[/bold green] {result['message']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
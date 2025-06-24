#!/usr/bin/env python3

import os
import sys
import requests
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Initialize rich console for better output formatting
console = Console()

def check_server(server_url="http://0.0.0.0:8000", max_retries=3):
    """Check if the server is running with retries"""
    console.print(f"Checking server at {server_url}...")
    
    for i in range(max_retries):
        try:
            response = requests.get(server_url, timeout=5)
            response.raise_for_status()
            console.print(f"[bold green]Server is running![/bold green]")
            return True
        except requests.exceptions.RequestException as e:
            console.print(f"[yellow]Attempt {i+1}/{max_retries}: Server not responding ({str(e)})[/yellow]")
            if i < max_retries - 1:
                console.print("Retrying in 2 seconds...")
                time.sleep(2)
    
    console.print("[bold red]Error:[/bold red] RAG server is not running. Please start it with 'python rag_server.py'")
    return False

def ingest_document(file_path, server_url="http://0.0.0.0:8000"):
    """Ingest a document into the RAG system"""
    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
        return None
    
    # Prepare form data
    files = {"document": open(file_path, "rb")}
    data = {
        "strategy": "paragraph",
        "chunk_size": "1000",
        "overlap": "100",
        "title": os.path.basename(file_path)
    }
    
    try:
        console.print(f"Ingesting document: [bold]{file_path}[/bold]")
        response = requests.post(f"{server_url}/ingest", files=files, data=data)
        response.raise_for_status()
        result = response.json()
        
        console.print(f"[bold green]Success:[/bold green] {result['message']}")
        console.print(f"Chunks count: {result['chunks_count']}")
        
        return result
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            console.print(f"Response: {e.response.text}")
        return None
    finally:
        files["document"].close()

def query_rag(query_text, server_url="http://0.0.0.0:8000"):
    """Query the RAG system"""
    try:
        console.print(f"\nQuerying: [bold]\"{query_text}\"[/bold]")
        response = requests.post(
            f"{server_url}/query",
            json={"query": query_text, "top_k": 3}
        )
        response.raise_for_status()
        result = response.json()
        
        # Display the answer
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
        
        # Display the sources
        if result.get("sources"):
            console.print("[bold]Sources:[/bold]")
            for i, source in enumerate(result["sources"]):
                console.print(f"[bold cyan]{i+1}.[/bold cyan] [yellow]{source['text'][:150]}...[/yellow]")
                console.print(f"   Score: {source['score']:.4f}")
                console.print()
        
        return result
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            console.print(f"Response: {e.response.text}")
        return None

def main():
    # Welcome message
    console.print(Panel.fit("[bold]RAG System Demo[/bold]", border_style="green"))
    
    # Check if server is running
    server_url = "http://0.0.0.0:8000"
    if not check_server(server_url):
        sys.exit(1)
    
    # Ingest README.md
    readme_path = "RAG_README.md"
    ingest_result = ingest_document(readme_path, server_url)
    
    if not ingest_result:
        sys.exit(1)
    
    # Sample queries
    queries = [
        "What is a RAG system?",
        "What are the components of this RAG system?",
        "How do I ingest a document using the client?",
        "What vector databases can I use with this system?"
    ]
    
    for query in queries:
        query_rag(query, server_url)
        console.print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()
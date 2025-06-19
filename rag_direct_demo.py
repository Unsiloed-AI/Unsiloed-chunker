#!/usr/bin/env python3

import os
import sys
import Unsiloed
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rag_system import InMemoryVectorStore, create_embedding, Agent
import asyncio

# Initialize rich console for better output formatting
console = Console()

def process_document(file_path, strategy="paragraph"):
    """Process a document using Unsiloed"""
    # Check if file exists
    if not os.path.exists(file_path):
        console.print(f"[bold red]Error:[/bold red] File {file_path} does not exist")
        return None
    
    try:
        console.print(f"Processing document: [bold]{file_path}[/bold]")
        result = Unsiloed.process_sync({
            "filePath": file_path,
            "strategy": strategy,
            "chunkSize": 1000,
            "overlap": 100
        })
        
        console.print(f"[bold green]Success:[/bold green] Document processed")
        console.print(f"Chunks count: {len(result.get('chunks', []))}")
        
        return result
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        return None

def ingest_to_vector_store(chunks, vector_store, source_name):
    """Add document chunks to vector store"""
    doc_ids = []
    
    console.print(f"Ingesting {len(chunks)} chunks to vector store...")
    
    for i, chunk in enumerate(chunks):
        # Create unique ID for the chunk
        doc_id = f"doc_{source_name}_{i}"
        doc_ids.append(doc_id)
        
        # Create embedding for the chunk
        text = chunk.get("text", "")
        embedding = create_embedding(text)
        
        # Add metadata
        metadata = {
            "source": source_name,
            "chunk_index": i
        }
        if "metadata" in chunk and "page" in chunk["metadata"]:
            metadata["page"] = chunk["metadata"]["page"]
        
        # Add to vector store
        vector_store.add_document(doc_id, text, embedding, metadata)
    
    console.print(f"[bold green]Success:[/bold green] Added {len(chunks)} chunks to vector store")
    return doc_ids

def query_agent(agent, query_text):
    """Query the agent"""
    console.print(f"\nQuerying: [bold]\"{query_text}\"[/bold]")
    
    result = agent.process_query_sync(query_text, top_k=3)
    
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

def main():
    # Welcome message
    console.print(Panel.fit("[bold]RAG System Direct Demo[/bold]", border_style="green"))
    
    # Initialize vector store
    vector_store = InMemoryVectorStore()
    
    # Process README.md
    readme_path = "RAG_README.md"
    result = process_document(readme_path)
    
    if not result:
        sys.exit(1)
    
    # Ingest to vector store
    ingest_to_vector_store(result.get("chunks", []), vector_store, os.path.basename(readme_path))
    
    # Initialize agent
    agent = Agent(vector_store)
    
    # Sample queries
    queries = [
        "What is a RAG system?",
        "What are the components of this RAG system?",
        "How do I ingest a document using the client?",
        "What vector databases can I use with this system?"
    ]
    
    for query in queries:
        query_agent(agent, query)
        console.print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    main()
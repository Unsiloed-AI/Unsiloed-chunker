#!/usr/bin/env python3

import os
import sys
import requests
import gradio as gr
import tempfile

# Server URL
SERVER_URL = "http://0.0.0.0:8000"

# Check if server is running
def check_server():
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.ConnectionError:
        return False

# Get list of documents
def get_documents():
    try:
        response = requests.get(f"{SERVER_URL}/documents")
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.ConnectionError:
        return []

# Ingest document
def ingest_document(file, strategy, chunk_size, overlap):
    if not file:
        return "Please select a file to upload"
    
    if not check_server():
        return "RAG server is not running. Please start it with 'python rag_server.py'"
    
    try:
        # Create form data
        files = {"file": (os.path.basename(file.name), open(file.name, "rb"))}
        data = {
            "strategy": strategy,
            "chunk_size": chunk_size,
            "overlap": overlap
        }
        
        # Send request
        response = requests.post(f"{SERVER_URL}/ingest", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            return f"Document ingested successfully! Document ID: {result.get('document_id')}\nChunks: {result.get('chunks_count')}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Query the RAG system
def query_rag(query):
    if not query.strip():
        return "Please enter a query"
    
    if not check_server():
        return "RAG server is not running. Please start it with 'python rag_server.py'"
    
    try:
        # Send request
        response = requests.post(
            f"{SERVER_URL}/query",
            json={"query": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No answer provided")
            
            # Format sources
            sources = result.get("sources", [])
            sources_text = "\n\n**Sources:**\n"
            for i, source in enumerate(sources):
                sources_text += f"**{i+1}.** {source['text'][:150]}...\n"
                sources_text += f"   Score: {source['score']:.4f}\n\n"
            
            return f"{answer}\n{sources_text if sources else ''}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Delete document
def delete_document(doc_id):
    if not doc_id:
        return "Please select a document to delete"
    
    if not check_server():
        return "RAG server is not running. Please start it with 'python rag_server.py'"
    
    try:
        # Send request
        response = requests.delete(f"{SERVER_URL}/documents/{doc_id}")
        
        if response.status_code == 200:
            return f"Document {doc_id} deleted successfully!"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Refresh document list
def refresh_documents():
    docs = get_documents()
    doc_ids = [doc["id"] for doc in docs]
    return gr.Dropdown.update(choices=doc_ids, value=doc_ids[0] if doc_ids else None)

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="RAG System") as app:
        gr.Markdown("# Agentic RAG System")
        
        with gr.Tab("Query"):
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(label="Query", placeholder="Enter your question here...")
                    query_button = gr.Button("Submit Query")
                
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
            
            query_button.click(query_rag, inputs=query_input, outputs=answer_output)
        
        with gr.Tab("Ingest Document"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload Document")
                    strategy_input = gr.Dropdown(
                        label="Chunking Strategy",
                        choices=["semantic", "fixed", "paragraph", "heading", "page"],
                        value="paragraph"
                    )
                    chunk_size_input = gr.Slider(
                        label="Chunk Size",
                        minimum=100,
                        maximum=2000,
                        value=1000,
                        step=100
                    )
                    overlap_input = gr.Slider(
                        label="Overlap",
                        minimum=0,
                        maximum=500,
                        value=100,
                        step=10
                    )
                    ingest_button = gr.Button("Ingest Document")
                
                with gr.Column():
                    ingest_output = gr.Textbox(label="Result")
            
            ingest_button.click(
                ingest_document,
                inputs=[file_input, strategy_input, chunk_size_input, overlap_input],
                outputs=ingest_output
            )
        
        with gr.Tab("Manage Documents"):
            with gr.Row():
                with gr.Column():
                    refresh_button = gr.Button("Refresh Document List")
                    doc_dropdown = gr.Dropdown(label="Select Document", choices=[])
                    delete_button = gr.Button("Delete Document")
                
                with gr.Column():
                    manage_output = gr.Textbox(label="Result")
            
            refresh_button.click(refresh_documents, inputs=None, outputs=doc_dropdown)
            delete_button.click(delete_document, inputs=doc_dropdown, outputs=manage_output)
        
        # Check server status on load
        if not check_server():
            gr.Warning("RAG server is not running. Please start it with 'python rag_server.py'")
    
    return app

if __name__ == "__main__":
    # Create and launch the interface
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
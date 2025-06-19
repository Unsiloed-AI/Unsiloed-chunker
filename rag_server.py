import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import Unsiloed
import tempfile
import shutil
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import uuid
from rag_system import vector_store, create_embedding, agent

app = FastAPI(title="Agentic RAG API", description="API for document ingestion, retrieval, and question answering")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response validation
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class DocumentMetadata(BaseModel):
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None

class DocumentResponse(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata

@app.get("/")
async def root():
    return {"message": "Welcome to Agentic RAG API", "docs": "/docs"}

@app.post("/ingest")
async def ingest_document(
    document: UploadFile = File(...),
    strategy: str = Form("paragraph"),
    chunk_size: int = Form(1000),
    overlap: int = Form(100),
    api_key: str = Form(None),
    title: str = Form(None),
    author: str = Form(None),
    date: str = Form(None)
):
    # Set OpenAI API key if provided
    original_api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif strategy == "semantic" and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key is required for semantic chunking")
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(document.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the document using Unsiloed
        result = Unsiloed.process_sync({
            "filePath": temp_file_path,
            "strategy": strategy,
            "chunkSize": chunk_size,
            "overlap": overlap
        })
        
        # Create document metadata
        base_metadata = {
            "source": document.filename,
            "title": title,
            "author": author,
            "date": date
        }
        
        # Add each chunk to the vector store
        doc_ids = []
        for i, chunk in enumerate(result.get("chunks", [])):
            # Create unique ID for the chunk
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            # Create embedding for the chunk
            text = chunk.get("text", "")
            embedding = create_embedding(text)
            
            # Add metadata specific to this chunk
            metadata = base_metadata.copy()
            metadata["chunk_index"] = i
            if "metadata" in chunk and "page" in chunk["metadata"]:
                metadata["page"] = chunk["metadata"]["page"]
            
            # Add to vector store
            vector_store.add_document(doc_id, text, embedding, metadata)
        
        return JSONResponse(content={
            "message": f"Document processed and ingested successfully",
            "document_ids": doc_ids,
            "chunks_count": len(result.get("chunks", [])),
            "strategy": strategy
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)
        # Restore original API key
        if api_key:
            if original_api_key:
                os.environ["OPENAI_API_KEY"] = original_api_key
            else:
                del os.environ["OPENAI_API_KEY"]

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Process the query using the agent
        result = await agent.process_query(request.query, request.top_k)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    # Return list of all documents in the vector store
    documents = []
    for doc_id in vector_store.documents:
        documents.append({
            "id": doc_id,
            "text": vector_store.documents[doc_id][:100] + "...",  # Preview only
            "metadata": vector_store.metadata[doc_id]
        })
    return JSONResponse(content={"documents": documents})

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    # Delete a document from the vector store
    if doc_id in vector_store.documents:
        del vector_store.documents[doc_id]
        del vector_store.embeddings[doc_id]
        del vector_store.metadata[doc_id]
        return JSONResponse(content={"message": f"Document {doc_id} deleted successfully"})
    else:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

if __name__ == "__main__":
    # Run the server
    uvicorn.run("rag_server:app", host="0.0.0.0", port=8000, reload=True)
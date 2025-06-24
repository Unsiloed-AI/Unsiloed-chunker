import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import Unsiloed
import tempfile
import shutil

app = FastAPI(title="Unsiloed API", description="API for document chunking using Unsiloed")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to Unsiloed API", "docs": "/docs"}

@app.post("/chunk")
async def chunk_document(
    document: UploadFile = File(...),
    strategy: str = Form("paragraph"),
    chunk_size: int = Form(1000),
    overlap: int = Form(100),
    api_key: str = Form(None)
):
    # Set OpenAI API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif strategy == "semantic" and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=400, detail="OpenAI API key is required for semantic chunking")
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(document.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Process the document
        result = Unsiloed.process_sync({
            "filePath": temp_file_path,
            "strategy": strategy,
            "chunkSize": chunk_size,
            "overlap": overlap
        })
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

if __name__ == "__main__":
    # Run the server
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
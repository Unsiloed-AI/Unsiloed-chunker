from fastapi import FastAPI
from Unsiloed.routes.chunking_routes import router as chunking_router
from Unsiloed.routes.root_routes import router as root_router
from Unsiloed.routes.agentic_rag_routes import router as agentic_rag_router

app = FastAPI()

app.include_router(chunking_router)
app.include_router(root_router)
app.include_router(agentic_rag_router)

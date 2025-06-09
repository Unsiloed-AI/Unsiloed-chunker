# Agentic RAG Retrieval System

This module implements an agentic RAG (Retrieval-Augmented Generation) system that can handle complex queries, including multi-hop queries and negation queries.

## Overview

Traditional RAG systems often struggle with complex queries that require multiple steps of reasoning or filtering out specific information. The agentic RAG system addresses these limitations by using a LangChain ReAct agent that can:

1. Break down complex multi-hop queries into simpler sub-queries
2. Handle negation by filtering out irrelevant information
3. Provide detailed reasoning for the retrieval process

## Architecture

The agentic RAG system consists of the following components:

### Core Components

- **AgenticRAG Class**: The main class that implements the agentic RAG system
- **Vector Store**: FAISS vector database for efficient similarity search
- **LLM**: OpenAI's GPT-4o for generating responses and reasoning
- **Agent Executor**: LangChain's AgentExecutor for orchestrating the retrieval process

### Tools

The agent has access to the following specialized tools:

1. **Search Tool**: For retrieving relevant documents based on a query
2. **Decompose Tool**: For breaking down complex queries into simpler sub-queries
3. **FilterNegation Tool**: For filtering out documents that contain negated concepts

## Query Processing Flow

### Multi-hop Queries

For multi-hop queries (e.g., "What are the key findings and their implications?"):

1. The agent identifies that the query requires multiple steps of reasoning
2. It uses the Decompose tool to break down the query into sub-queries
   - Sub-query 1: "What are the key findings mentioned in the document?"
   - Sub-query 2: "What implications are associated with these findings?"
3. It searches for information related to each sub-query
4. It combines the results to generate a comprehensive answer

### Negation Queries

For negation queries (e.g., "What topics are discussed excluding financial data?"):

1. The agent identifies that the query contains negation
2. It performs an initial search to retrieve relevant documents
3. It uses the FilterNegation tool to filter out documents related to financial data
4. It generates an answer based on the filtered documents

### Simple Queries

For simple queries (e.g., "What is the main topic of this document?"):

1. The agent identifies that the query is straightforward
2. It directly searches for relevant information
3. It generates an answer based on the retrieved documents

## Usage

### Adding Documents

```python
from Unsiloed.services.agentic_rag.core import AgenticRAG

# Initialize the AgenticRAG system
rag = AgenticRAG()

# Add documents
documents = [
    {"text": "Document content 1", "metadata": "value1"},
    {"text": "Document content 2", "metadata": "value2"}
]
rag.add_documents(documents)
```

### Processing Queries

```python
# Process a query
result = rag.query("What are the key findings and their implications?")

# Access the results
answer = result["answer"]
reasoning = result["reasoning"]
query_type = result["query_type"]
```

## API Endpoints

The agentic RAG system exposes the following API endpoints:

- **POST /agentic_rag/query**: Process a query using provided document chunks
- **POST /agentic_rag/process_chunks**: Process chunks from the chunking service and query
- **POST /agentic_rag/reset**: Reset the agentic RAG system

## Integration with Chunking Service

The agentic RAG system integrates seamlessly with the existing chunking service:

1. Use the chunking service to chunk a document
2. Pass the chunks to the agentic RAG system
3. Process queries using the agentic RAG system

Example:

```python
import requests

# First, chunk a document
with open("document.pdf", "rb") as f:
    files = {"document_file": ("document.pdf", f)}
    data = {"strategy": "semantic"}
    chunking_response = requests.post("http://localhost:8000/chunking", files=files, data=data)

chunks_result = chunking_response.json()

# Then, process a query using the agentic RAG system
query_data = {
    "query": "What are the key findings and their implications?",
    "chunks_result": chunks_result
}

rag_response = requests.post("http://localhost:8000/agentic_rag/process_chunks", json=query_data)
rag_result = rag_response.json()
``` 
# Agentic RAG Retrieval System

This is an agentic RAG (Retrieval-Augmented Generation) system built on top of the Unsiloed document chunking library. The system provides document ingestion, vector storage, semantic search, and AI-powered question answering capabilities.

## Features

- **Document Processing**: Leverages Unsiloed's powerful document chunking capabilities with multiple strategies (semantic, fixed, paragraph, heading, page)
- **Vector Storage**: In-memory vector database for storing document chunks and their embeddings
- **Semantic Search**: Find the most relevant document chunks for a given query using cosine similarity
- **Agentic Responses**: Generate contextual answers to questions using retrieved documents and OpenAI's language models
- **API Server**: FastAPI-based server with endpoints for document ingestion, querying, and management
- **Command-line Client**: Easy-to-use CLI for interacting with the RAG system

## Components

1. **rag_system.py**: Core components including vector store, embedding creation, and agent logic
2. **rag_server.py**: FastAPI server with endpoints for document operations and querying
3. **rag_client.py**: Command-line client for interacting with the RAG API

## Prerequisites

- Python 3.8+
- OpenAI API key (for semantic chunking and AI responses)
- Unsiloed document chunker

## Installation

1. Make sure you have the Unsiloed chunker installed:

```bash
pip install -e .
```

2. Install additional dependencies:

```bash
pip install openai python-dotenv rich
```

3. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Starting the Server

Run the RAG server:

```bash
python rag_server.py
```

The server will start on http://localhost:8000 by default. You can access the API documentation at http://localhost:8000/docs.

### Using the Command-line Client

The `rag_client.py` script provides a convenient way to interact with the RAG system.

#### Ingesting Documents

```bash
python rag_client.py ingest path/to/document.pdf --strategy paragraph --title "Document Title" --author "Author Name"
```

Options:
- `--strategy`: Chunking strategy (semantic, fixed, paragraph, heading, page)
- `--chunk-size`: Size of chunks (default: 1000)
- `--overlap`: Overlap between chunks (default: 100)
- `--api-key`: OpenAI API key (required for semantic chunking if not set in environment)
- `--title`: Document title
- `--author`: Document author
- `--date`: Document date

#### Querying the System

```bash
python rag_client.py query "What is the main topic of the document?"
```

Options:
- `--top-k`: Number of top results to return (default: 5)

#### Listing Documents

```bash
python rag_client.py list
```

#### Deleting Documents

```bash
python rag_client.py delete document-id
```

### API Endpoints

- `GET /`: Welcome message
- `POST /ingest`: Ingest a document
- `POST /query`: Query the system
- `GET /documents`: List all documents
- `DELETE /documents/{doc_id}`: Delete a document

## Architecture

### Document Ingestion Flow

1. User uploads a document through the API or client
2. Document is processed using Unsiloed chunker with the specified strategy
3. Each chunk is converted to an embedding using OpenAI's embedding model
4. Chunks and embeddings are stored in the vector database with metadata

### Query Flow

1. User submits a query through the API or client
2. Query is converted to an embedding
3. Vector database finds the most similar document chunks
4. Retrieved chunks are used as context for the AI agent
5. Agent generates a response based on the retrieved information
6. Response and source documents are returned to the user

## Extending the System

### Using a Different Vector Database

The current implementation uses a simple in-memory vector store. For production use, you might want to replace it with a more robust solution like:

- Pinecone
- Weaviate
- Milvus
- Qdrant
- Chroma

To do this, modify the `InMemoryVectorStore` class in `rag_system.py` to interface with your preferred vector database.

### Customizing the Agent

The agent's behavior can be customized by modifying the `Agent` class in `rag_system.py`. You can change the prompt, model, or add additional reasoning steps.

## Limitations

- The in-memory vector store is not persistent and will lose data when the server restarts
- The system currently only supports text-based queries
- Performance may be limited for very large document collections

## License

This project is licensed under the same license as the Unsiloed chunker.
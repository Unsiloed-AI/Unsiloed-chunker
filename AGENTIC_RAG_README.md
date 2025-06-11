# Agentic RAG Implementation Test Guide

This guide provides instructions for testing the agentic RAG (Retrieval-Augmented Generation) system implementation.

## Prerequisites

- Python 3.8+
- OpenAI API key
- PDF document for testing

## Setup

1. Make sure you have all required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Alternatively, create a `.env` file with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Running the Test Script

The `test_agentic_rag.py` script demonstrates the agentic RAG implementation by:
1. Indexing a PDF document
2. Running a simple query
3. Running a multi-hop query
4. Running a negation query
5. Cleaning up (optional)

### Usage

```bash
python test_agentic_rag.py path/to/your/document.pdf [--keep]
```

Options:
- `path/to/your/document.pdf`: Path to the PDF document you want to test
- `--keep`: Optional flag to keep the document in the database after testing

### Example

```bash
python test_agentic_rag.py sample.pdf
```

## Testing with API Endpoints

You can also test the implementation using the FastAPI endpoints:

1. Start the FastAPI server:
   ```bash
   uvicorn Unsiloed.main:app --reload
   ```

2. Open the Swagger UI in your browser:
   ```
   http://localhost:8000/docs
   ```

3. Test the following endpoints:
   - POST `/retrieval/index`: Index a document
   - POST `/retrieval/query`: Query the indexed documents
   - DELETE `/retrieval/document/{document_id}`: Delete a document

## Understanding the Output

The test script will provide detailed output for each step:

1. **Indexing**: Shows document ID and number of chunks created
2. **Simple Query**: Shows the query type and top chunks with similarity scores
3. **Multi-hop Query**: Shows how the query is broken down into sub-queries and the synthesized answer
4. **Negation Query**: Shows how negation is handled and the final answer

## Features Demonstrated

- **Query Analysis**: Automatic detection of query types (simple, multi-hop, negation)
- **Query Decomposition**: Breaking down complex queries into simpler sub-queries
- **Semantic Search**: Finding relevant document chunks based on embedding similarity
- **Result Synthesis**: Combining information from multiple chunks into a coherent answer

## Troubleshooting

- If you encounter an error about missing the OpenAI API key, ensure it's properly set in the environment or `.env` file
- If the SQLite database causes issues, delete the `vector_store.db` file and try again
- For chunking errors, try a different chunking strategy (e.g., "fixed" instead of "semantic") 
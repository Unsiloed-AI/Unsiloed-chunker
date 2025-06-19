# Document Processor - Setup and Deployment Guide

This guide provides step-by-step instructions for setting up, running, and deploying the Document Processor application.

## Table of Contents

- [Local Setup](#local-setup)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

## Local Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Unsiloed-AI/Unsiloed-chunker.git
   cd Unsiloed-chunker
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Set your OpenAI API key (required for semantic chunking):
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY="your-api-key-here"
   
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your-api-key-here
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your-api-key-here"
   ```

## Running the Application

### Running the Python Package

You can use the Unsiloed package directly in your Python code:

```python
import DocumentProcessor

# Example using paragraph chunking (doesn't require OpenAI API key)
result = Unsiloed.process_sync({
    "filePath": "path/to/document.pdf",
    "strategy": "paragraph"
})

# Print the first chunk
if result["chunks"]:
    print(f"Total chunks: {result['total_chunks']}")
    print(f"First chunk: {result['chunks'][0]['text'][:100]}...")
```

### Running the API Server

To start the FastAPI server:

```bash
python server.py
```

The API will be available at http://localhost:8000, and the interactive documentation at http://localhost:8000/docs.

### Using the Client Script

The client.py script provides a command-line interface for interacting with the API:

```bash
# Basic usage (paragraph chunking)
python client.script path/to/document.pdf

# Fixed size chunking
python client.py path/to/document.pdf --strategy fixed --chunk-size 500 --overlap 50

# Semantic chunking (requires OpenAI API key)
python client.py path/to/document.pdf --strategy semantic --api-key your-api-key-here

# Save results to a file
python client.py path/to/document.pdf --save results.json
```

## API Usage

### Endpoints

- `GET /`: Welcome message and API information
- `POST /chunk`: Process a document file with the specified chunking strategy

### Example API Request

Using curl:

```bash
curl -X POST http://localhost:8000/chunk \
  -F "document=@path/to/document.pdf" \
  -F "strategy=paragraph" \
  -F "chunk_size=1000" \
  -F "overlap=100"
```

Using Python requests:

```python
import requests

url = "http://localhost:8000/chunk"
files = {"document": open("path/to/document.pdf", "rb")}
data = {"strategy": "paragraph", "chunk_size": "1000", "overlap": "100"}

response = requests.post(url, files=files, data=data)
result = response.json()
```

## Docker Deployment

### Building and Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t unsiloed-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 -e OPENAI_API_KEY="your-api-key-here" unsiloed-api
   ```

### Using Docker Compose

1. Set your OpenAI API key in the environment:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Start the services:
   ```bash
   docker-compose up
   ```

## Production Deployment

For detailed production deployment instructions, see the DEPLOYMENT document.

### Key Considerations for Production

1. **Security**:
   - Use environment variables for sensitive information
   - Implement proper authentication for API endpoints
   - Use HTTPS in production

2. **Scaling**:
   - Use a load balancer for high-traffic applications
   - Implement caching for frequently requested documents
   - Monitor resource usage and adjust instance sizes as needed

3. **Monitoring**:
   - Set up logging to track API usage and errors
   - Implement health checks for API endpoints
   - Use monitoring tools to track performance metrics

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**:
   - Ensure the API key is correctly set in the environment variables
   - Check if the API key has sufficient permissions

2. **Memory Issues**:
   - Large documents may require more memory
   - Consider increasing the memory allocation for the application

3. **Performance Issues**:
   - Adjust the number of workers based on CPU cores available
   - Consider using a more powerful instance for processing large documents

### Getting Help

If you encounter any issues, please open an issue on the [GitHub repository](https://github.com/Unsiloed-AI/Unsiloed-chunker/issues).
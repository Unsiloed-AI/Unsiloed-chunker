# 📄 Unsiloed AI Document Data extractor

A super simple way to extract text from documents for intelligent document processing, extraction, and chunking with multi-threaded processing capabilities.

## 🚀 Features

### 📊 Document Chunking

- **Supported File Types**: PDF, DOCX, PPTX
- **Chunking Strategies**:
  - **Fixed Size**: Splits text into chunks of specified size with optional overlap
  - **Page-based**: Splits PDF by pages (PDF only, falls back to paragraph for other file types)
  - **Semantic**: Uses LLM to identify meaningful semantic chunks
  - **Paragraph**: Splits text by paragraphs
  - **Heading**: Splits text by identified headings

### 🤖 Model Provider Support

- **OpenAI**: GPT-4 and other OpenAI models
- **Anthropic**: Claude models
- **HuggingFace**: Local inference with transformers
- **Configurable**: Easy to add new model providers

## 🔧 Technical Details

### 🧠 LLM Integration

- Supports multiple model providers:
  - OpenAI GPT-4 for semantic chunking
  - Anthropic Claude models
  - HuggingFace models for local inference
- Handles authentication via API keys from environment variables
- Implements automatic retries and timeout handling
- Provides structured JSON output for semantic chunks

### 🔄 Parallel Processing

- Multi-threaded processing for improved performance
- Parallel page extraction from PDFs
- Distributes processing of large documents across multiple threads

### 📝 Document Processing

- Extracts text from PDF, DOCX, and PPTX files
- Handles image encoding for vision-based models
- Generates extraction prompts for structured data extraction

## ⚙️ Configuration

### Environmental Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key (for Claude models)

## 🛑 Constraints & Limitations

### File Handling

- Temporary files are created during processing and deleted afterward
- Files are processed in-memory where possible

### Text Processing

- Long text (>25,000 characters) is automatically split and processed in parallel for semantic chunking
- Maximum token limit of 4000 for model responses

### API Constraints

- Request timeout set to 60 seconds
- Maximum of 3 retries for API calls

## 📋 Request Parameters

### Document Chunking Endpoint

- `document_file`: The document file to process (PDF, DOCX, PPTX)
- `strategy`: Chunking strategy to use (default: "semantic")
  - Options: "fixed", "page", "semantic", "paragraph", "heading"
- `chunk_size`: Size of chunks for fixed strategy in characters (default: 1000)
- `overlap`: Overlap size for fixed strategy in characters (default: 100)
- `model_provider`: Type of model provider to use (default: "openai")
  - Options: "openai", "anthropic", "huggingface"
- `model_config`: Additional configuration for the model provider (optional)

## 📦 Installation

### Using pip

```bash
pip install unsiloed
```

### Requirements

Unsiloed requires Python 3.8 or higher and has the following dependencies:

- openai
- anthropic
- transformers
- torch
- PyPDF2
- python-docx
- python-pptx
- fastapi
- python-multipart

## 🔑 Environment Setup

Before using Unsiloed, set up your API keys:

### Using environment variables

```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Windows (Command Prompt)
set OPENAI_API_KEY=your-openai-api-key
set ANTHROPIC_API_KEY=your-anthropic-api-key

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Using a .env file

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Then in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env
```

## 🔍 Usage Example

### Python

```python
import os
import Unsiloed

# Example 1: Semantic chunking with OpenAI (default)
result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "credentials": {
        "apiKey": os.environ.get("OPENAI_API_KEY")
    },
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Example 2: Semantic chunking with Anthropic Claude
claude_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "credentials": {
        "apiKey": os.environ.get("ANTHROPIC_API_KEY")
    },
    "strategy": "semantic",
    "modelProvider": "anthropic",
    "modelConfig": {
        "model": "claude-3-opus-20240229"
    }
})

# Example 3: Semantic chunking with HuggingFace
hf_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "strategy": "semantic",
    "modelProvider": "huggingface",
    "modelConfig": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2"
    }
})

# Example 4: Fixed-size chunking
fixed_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "strategy": "fixed",
    "chunkSize": 1500,
    "overlap": 150
})

# Example 5: Page-based chunking (PDF only)
page_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "strategy": "page"
})

# Example 6: Paragraph chunking
paragraph_result = Unsiloed.process_sync({
    "filePath": "./document.docx",
    "strategy": "paragraph"
})

# Example 7: Heading chunking
heading_result = Unsiloed.process_sync({
    "filePath": "./presentation.pptx",
    "strategy": "heading"
})
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🌐 Community and Support

### Join the Community

- **GitHub Discussions**: For questions, ideas, and discussions
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For contributing to the codebase

### Staying Updated

- **Star** the repository to show support
- **Watch** for notification on new releases

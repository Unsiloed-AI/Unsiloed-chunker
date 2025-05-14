# 📄 Unsiloed AI Document Data extractor

A super simple way to extract text from documents for  for intelligent document processing, extraction, and chunking with multi-threaded processing capabilities.

## 🚀 Features

### 📊 Document Chunking
- **Supported File Types**:
  - **Document Formats**: PDF, DOCX, DOC, TXT, RTF, EPUB
  - **Spreadsheet Formats**: XLSX, XLS, ODS
  - **Presentation Formats**: PPTX, ODP
  - **OpenDocument Formats**: ODT, ODS, ODP
- **Chunking Strategies**:
  - **Fixed Size**: Splits text into chunks of specified size with optional overlap
  - **Page-based**: Splits PDF by pages (PDF only, falls back to paragraph for other file types)
  - **Semantic**: Uses Multi-Modal Model to identify meaningful semantic chunks
  - **Paragraph**: Splits text by paragraphs
  - **Heading**: Splits text by identified headings

## 🔧 Technical Details

### 🧠 Multiple Model Support
- Supports multiple LLM providers:
  - **OpenAI**: GPT-4o and other OpenAI models
  - **Anthropic**: Claude models
  - **Hugging Face**: Hosted models like Mistral
  - **Local**: Self-hosted models via llama.cpp
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
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `HUGGINGFACE_API_KEY`: Your Hugging Face API key
- `LOCAL_MODEL_PATH`: Path to your local LLM model
- `UNSILOED_MODEL_PROVIDER`: Default model provider to use (openai, anthropic, huggingface, local)

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
- Different models have different capabilities and limitations
  - OpenAI models: Best overall quality for semantic chunking
  - Anthropic models: Good alternative with similar capabilities
  - Hugging Face models: Varies by model, may require specific model selection
  - Local models: Performance depends on hardware and model size

## 📋 Request Parameters

### Document Chunking Endpoint
- `document_file`: The document file to process (PDF, DOCX, PPTX, DOC, XLSX, XLS, ODT, ODS, ODP, TXT, RTF, EPUB)
- `strategy`: Chunking strategy to use (default: "semantic")
  - Options: "fixed", "page", "semantic", "paragraph", "heading"
- `chunk_size`: Size of chunks for fixed strategy in characters (default: 1000)
- `overlap`: Overlap size for fixed strategy in characters (default: 100)


## 📦 Installation

### Using pip
```bash
# Basic installation with OpenAI support
pip install unsiloed

# Install with all model providers
pip install unsiloed[all]

# Install with specific model providers
pip install unsiloed[anthropic]  # For Anthropic Claude support
pip install unsiloed[huggingface]  # For Hugging Face models support
pip install unsiloed[local]  # For local LLM support via llama.cpp
```

### Requirements
Unsiloed requires Python 3.8 or higher and has the following dependencies:
- openai
- PyPDF2
- python-docx
- python-pptx
- fastapi
- python-multipart
- docx2txt (for DOC files)
- openpyxl (for XLSX files)
- xlrd (for XLS files)
- odfpy (for ODT, ODS, ODP files)
- ebooklib and beautifulsoup4 (for EPUB files)
- striprtf (for RTF files)

Optional dependencies based on model provider:
- anthropic (for Claude models)
- huggingface_hub (for Hugging Face models)
- llama-cpp-python (for local LLM models)

## 🔑 Environment Setup

Before using Unsiloed, set up your API keys for the model providers you want to use:

### Using environment variables
```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export HUGGINGFACE_API_KEY="your-huggingface-api-key-here"
export LOCAL_MODEL_PATH="/path/to/your/local/model.gguf"
export UNSILOED_MODEL_PROVIDER="openai"  # Default model provider

# Windows (Command Prompt)
set OPENAI_API_KEY=your-openai-api-key-here
set ANTHROPIC_API_KEY=your-anthropic-api-key-here
set HUGGINGFACE_API_KEY=your-huggingface-api-key-here
set LOCAL_MODEL_PATH=C:\path\to\your\local\model.gguf
set UNSILOED_MODEL_PROVIDER=openai

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-openai-api-key-here"
$env:ANTHROPIC_API_KEY="your-anthropic-api-key-here"
$env:HUGGINGFACE_API_KEY="your-huggingface-api-key-here"
$env:LOCAL_MODEL_PATH="C:\path\to\your\local\model.gguf"
$env:UNSILOED_MODEL_PROVIDER="openai"
```

### Using a .env file
Create a `.env` file in your project directory:
```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here
LOCAL_MODEL_PATH=/path/to/your/local/model.gguf
UNSILOED_MODEL_PROVIDER=openai
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
    "modelProvider": "openai",
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Print the result
print(result)

# Example 2: Semantic chunking with Anthropic Claude
claude_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "credentials": {
        "anthropicApiKey": os.environ.get("ANTHROPIC_API_KEY")
    },
    "modelProvider": "anthropic",
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Example 3: Semantic chunking with Hugging Face
hf_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "credentials": {
        "huggingfaceApiKey": os.environ.get("HUGGINGFACE_API_KEY")
    },
    "modelProvider": "huggingface",
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Example 4: Semantic chunking with local LLM
local_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "credentials": {
        "localModelPath": os.environ.get("LOCAL_MODEL_PATH")
    },
    "modelProvider": "local",
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Example 5: Fixed-size chunking (model-agnostic)
fixed_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "strategy": "fixed",
    "chunkSize": 1500,
    "overlap": 150
})

# Example 6: Page-based chunking (PDF only, model-agnostic)
page_result = Unsiloed.process_sync({
    "filePath": "./test.pdf",
    "strategy": "page"
})

# Example 7: Paragraph chunking (model-agnostic)
paragraph_result = Unsiloed.process_sync({
    "filePath": "./document.docx",
    "strategy": "paragraph"
})

# Example 8: Heading chunking (model-agnostic)
heading_result = Unsiloed.process_sync({
    "filePath": "./presentation.pptx",
    "strategy": "heading"
})
```

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- git

### Setting Up Local Development Environment

1. Clone the repository:
```bash
git clone https://github.com/Unsiloed-opensource/Unsiloed.git
cd Unsiloed
```

2. Create a virtual environment:
```bash
# Using venv
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
# Create a .env file with your API keys
cat > .env << EOL
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here
LOCAL_MODEL_PATH=/path/to/your/local/model.gguf
UNSILOED_MODEL_PROVIDER=openai
EOL
```

5. Run the FastAPI server locally:
```bash
uvicorn Unsiloed.app:app --reload
```

6. Access the API documentation:
Open your browser and go to `http://localhost:8000/docs`



## 👨‍💻 Contributing

We welcome contributions to Unsiloed! Here's how you can help:

### Setting Up Development Environment

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/Unsiloed.git
cd Unsiloed
```

2. Install development dependencies:
```bash
pip install -r requirements.txt
```


### Making Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and write tests if applicable


4. Commit your changes:
```bash
git commit -m "Add your meaningful commit message here"
```

5. Push to your fork:
```bash
git push origin feature/your-feature-name
```

6. Create a Pull Request from your fork to the main repository

### Code Style and Standards

- We follow PEP 8 for Python code style
- Use type hints where appropriate
- Document functions and classes with docstrings
- Write tests for new features


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Community and Support

### Join the Community

- **GitHub Discussions**: For questions, ideas, and discussions
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For contributing to the codebase


### Staying Updated

- **Star** the repository to show support
- **Watch** for notification on new releases

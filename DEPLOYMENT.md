# Document Processor Deployment Guide

This guide provides instructions for deploying the Document Processor API in various environments.

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

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
   export OPENAI_API_KEY="your-api-key-here"  # On Windows: set OPENAI_API_KEY=your-api-key-here
   ```

5. Run the server:
   ```bash
   python server.py
   ```

6. Access the API documentation at http://localhost:8000/docs

## Docker Deployment

### Prerequisites

- Docker
- Docker Compose (optional)

### Using Docker

1. Create a Dockerfile in the project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY . /app/

RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Build and run the Docker image:

```bash
docker build -t unsiloed-api .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-api-key-here" unsiloed-api
```

### Using Docker Compose

1. Create a docker-compose.yaml file:

```yaml
version: '3'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

2. Run with Docker Compose:

```bash
docker-compose up
```

## Cloud Deployment

### Deploying to Cloud Platform

1. Create a process file in the project root:

```
web: uvicorn server:app --host=0.0.0.0 --port=${PORT:-8000}
```

2. Deploy to cloud platform:

```bash
heroku create unsiloed-api
heroku config:set OPENAI_API_KEY="your-api-key-here"
git push heroku main
```

### Deploying to AWS Elastic Beanstalk

1. Install the EB CLI:

```bash
pip install awsebcli
```

2. Initialize EB application:

```bash
eb init -p python-3.8 unsiloed-api
```

3. Create an environment and deploy:

```bash
eb create unsiloed-api-env
eb setenv OPENAI_API_KEY="your-api-key-here"
```

## Production Considerations

### Security

- Use environment variables for sensitive information like API keys
- Implement proper authentication for the API endpoints
- Consider using HTTPS in production

### Scaling

- For high-traffic applications, consider using a load balancer
- Implement caching for frequently requested documents
- Monitor resource usage and adjust instance sizes as needed

### Monitoring

- Set up logging to track API usage and errors
- Implement health checks for the API endpoints
- Use monitoring tools to track performance metrics

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   - Ensure the API key is correctly set in the environment variables
   - Check if the API key has sufficient permissions

2. **Memory Issues**
   - Large documents may require more memory
   - Consider increasing the memory allocation for the application

3. **Performance Issues**
   - Adjust the number of workers based on CPU cores available
   - Consider using a more powerful instance for processing large documents
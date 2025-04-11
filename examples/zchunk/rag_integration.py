#!/usr/bin/env python3
"""
Advanced example demonstrating zChunk integration with a RAG system.
This example uses a simple mock vector database to show the workflow.
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import random

# Add the parent directory to sys.path for imports to work from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from zchunk import LlamaChunker, ChunkerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class MockEmbeddingModel:
    """Simulate an embedding model that creates vector representations of text."""

    def __init__(self, dimension: int = 384):
        """Initialize the embedding model.

        Args:
            dimension: Dimensionality of the embedding vectors
        """
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        """Create a simple deterministic embedding for text.

        In a real system, you would use a proper embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # This is just a mock - in reality you would use a proper embedding model
        # like SentenceTransformers, OpenAI Embeddings, etc.
        random.seed(hash(text) % 10000)  # Use text hash for deterministic results
        return [random.uniform(-1, 1) for _ in range(self.dimension)]


class MockVectorDB:
    """Simulate a vector database for storing and retrieving chunks."""

    def __init__(self, embedding_model: MockEmbeddingModel):
        """Initialize the vector database.

        Args:
            embedding_model: Model used to create embeddings
        """
        self.embedding_model = embedding_model
        self.documents: List[Dict[str, Any]] = []

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the vector database.

        Args:
            text: Document text
            metadata: Additional metadata about the document
        """
        embedding = self.embedding_model.embed_text(text)
        self.documents.append(
            {"text": text, "embedding": embedding, "metadata": metadata or {}}
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for documents similar to the query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents with similarity scores
        """
        if not self.documents:
            return []

        query_embedding = self.embedding_model.embed_text(query)

        # Calculate cosine similarity (simplified)
        results = []
        for doc in self.documents:
            # Calculate dot product (simplified cosine similarity)
            similarity = sum(a * b for a, b in zip(query_embedding, doc["embedding"]))
            results.append(
                {"text": doc["text"], "metadata": doc["metadata"], "score": similarity}
            )

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top-k results
        return results[:top_k]


def process_document(filepath: str) -> None:
    """Process a document for RAG using zChunk.

    Args:
        filepath: Path to the document file
    """
    # Load document from file
    with open(filepath, "r") as f:
        document_text = f.read()

    print(f"Processing document ({len(document_text)} characters)...")

    # Initialize chunker with custom configuration
    config = ChunkerConfig(
        section_size=2000,
        overlap=100,
    )
    chunker = LlamaChunker(config)

    # Chunk the document
    print("Chunking document...")
    result = chunker.chunk_text(document_text)

    print(f"Created {len(result.big_chunks)} chunks")

    # Initialize mock components
    embedding_model = MockEmbeddingModel()
    vector_db = MockVectorDB(embedding_model)

    # Add chunks to vector database
    print("Adding chunks to vector database...")
    for i, chunk in enumerate(result.big_chunks):
        vector_db.add_document(
            text=chunk,
            metadata={
                "chunk_id": i,
                "source": filepath,
                "chunk_type": "big",
                "char_length": len(chunk),
            },
        )

    # Demonstrate RAG retrieval with a sample query
    sample_queries = [
        "What is machine learning?",
        "Explain supervised learning",
        "What are the applications of machine learning?",
    ]

    print("\nTesting RAG retrieval with sample queries:")
    for query in sample_queries:
        print(f"\nQuery: {query}")

        # Search for relevant chunks
        matching_chunks = vector_db.search(query, top_k=2)

        print(f"Found {len(matching_chunks)} relevant chunks:")
        for i, chunk in enumerate(matching_chunks):
            preview = (
                chunk["text"][:100] + "..."
                if len(chunk["text"]) > 100
                else chunk["text"]
            )
            print(f"  {i + 1}. Score: {chunk['score']:.3f}")
            print(f"     {preview}")

    # Save chunks to a file
    output_path = f"{Path(filepath).stem}_chunks.json"
    result.save_to_file(output_path)
    print(f"Chunks saved to {output_path}")


def main():
    """Run the RAG integration example."""

    # Create a simple document if not provided
    example_doc = "example_document.md"

    if not os.path.exists(example_doc):
        print(f"Creating example document {example_doc}...")
        with open(example_doc, "w") as f:
            f.write("""# Machine Learning Overview

Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. 
It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

### Supervised Learning

Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. 
The goal is to approximate the mapping function so well that when you have new input data (X), you can predict the output variables (Y) for that data.
It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.

Common supervised learning algorithms include:
- Linear regression
- Logistic regression
- Decision trees
- Random forests
- Support vector machines
- Neural networks

### Unsupervised Learning

Unsupervised learning is where you only have input data (X) and no corresponding output variables. 
The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.
These are called unsupervised learning because there is no correct answers and there is no teacher.

Common unsupervised learning algorithms include:
- k-means clustering
- Hierarchical clustering
- Principal component analysis
- Autoencoders

### Reinforcement Learning

Reinforcement learning is a type of machine learning where an agent learns to behave in an environment by performing actions and seeing the results.
The agent learns without intervention from a human by maximizing its reward and minimizing its penalty.

## Applications of Machine Learning

Machine learning has numerous applications across various domains:

1. **Computer Vision**: Image recognition, object detection, and facial recognition.
2. **Natural Language Processing**: Translation, sentiment analysis, and text generation.
3. **Healthcare**: Disease prediction, medical image analysis, and personalized medicine.
4. **Finance**: Fraud detection, algorithmic trading, and credit scoring.
5. **Autonomous Vehicles**: Self-driving cars and drones.
6. **Recommendation Systems**: Product recommendations, content suggestions, and personalized marketing.
""")

    # Process the document
    process_document(example_doc)


if __name__ == "__main__":
    main()

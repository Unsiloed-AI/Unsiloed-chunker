#!/usr/bin/env python3
"""
Example demonstrating zChunk with HTML content.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path for imports to work from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from zchunk import LlamaChunker, ChunkerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Run an example of the zChunk library with HTML content."""

    # Sample HTML to chunk
    sample_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Machine Learning Overview</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            header { background-color: #f4f4f4; padding: 20px; text-align: center; }
            .container { max-width: 800px; margin: auto; }
            h1, h2, h3 { color: #333; }
            section { margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <header>
            <h1>Introduction to Machine Learning</h1>
        </header>
        <div class="container">
            <section>
                <p>Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.</p>
            </section>
            <section>
                <h2>Types of Machine Learning</h2>
                <p>There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.</p>
                
                <h3>Supervised Learning</h3>
                <p>Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. The goal is to approximate the mapping function so well that when you have new input data (X), you can predict the output variables (Y) for that data.</p>
                
                <h3>Unsupervised Learning</h3>
                <p>Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.</p>
                
                <h3>Reinforcement Learning</h3>
                <p>Reinforcement learning is a type of machine learning where an agent learns to behave in an environment by performing actions and seeing the results. The agent learns without intervention from a human by maximizing its reward and minimizing its penalty.</p>
            </section>
            <section>
                <h2>Applications</h2>
                <ul>
                    <li>Computer vision</li>
                    <li>Natural language processing</li>
                    <li>Recommendation systems</li>
                    <li>Autonomous vehicles</li>
                    <li>Fraud detection</li>
                </ul>
            </section>
        </div>
        <footer>
            <p>&copy; 2023 Machine Learning Example</p>
        </footer>
    </body>
    </html>
    """

    print("Initializing chunker...")

    # Create a configuration for HTML content
    config = ChunkerConfig(
        section_size=2000,
        overlap=100,
        logprob_threshold=-6.0,  # Slightly higher threshold for HTML
    )

    # Initialize the chunker
    chunker = LlamaChunker(config)

    print("Chunking HTML content...")

    # Process the HTML
    result = chunker.chunk_text(sample_html)

    # Display results
    print(
        f"\nFound {len(result.big_chunks)} big chunks and {len(result.small_chunks)} small chunks.\n"
    )

    print("HTML Big Chunks:")
    for i, chunk in enumerate(result.big_chunks):
        # Show a preview of each chunk
        preview = chunk[:60] + "..." if len(chunk) > 60 else chunk
        print(f"Chunk {i + 1}: {preview}")

    # Example of how you might use these chunks for RAG
    print("\nExample RAG preparation:")
    for i, chunk in enumerate(result.big_chunks):
        # In a real application, you would embed each chunk
        print(
            f"Embedding chunk {i + 1} ({len(chunk)} characters, starts with: {chunk[:30]}...)"
        )

    # Save results
    output_path = "html_chunking_results.json"
    result.save_to_file(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

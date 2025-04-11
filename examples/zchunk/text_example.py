#!/usr/bin/env python3
"""
Simple example demonstrating the zChunk library.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path for imports to work from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from Unsiloed.utils.zchunk import LlamaChunker, ChunkerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Run a simple example of the zChunk library."""

    # Sample text to chunk
    sample_text = """
    # Introduction to Machine Learning
    
    Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.
    
    ## Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    ### Supervised Learning
    
    Supervised learning is where you have input variables (X) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. 
    The goal is to approximate the mapping function so well that when you have new input data (X), you can predict the output variables (Y) for that data.
    It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process.
    
    ### Unsupervised Learning
    
    Unsupervised learning is where you only have input data (X) and no corresponding output variables. 
    The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.
    These are called unsupervised learning because there is no correct answers and there is no teacher.
    
    ### Reinforcement Learning
    
    Reinforcement learning is a type of machine learning where an agent learns to behave in an environment by performing actions and seeing the results.
    The agent learns without intervention from a human by maximizing its reward and minimizing its penalty.
    """

    print("Initializing chunker...")

    # Create a simple configuration
    config = ChunkerConfig(
        # Use defaults for model loading
        section_size=1000,  # Process in smaller sections for this example
        overlap=50,
        logprob_threshold=-6.5,
    )

    # Initialize the chunker
    chunker = LlamaChunker(config)

    print("Chunking text...")

    # Process the text
    result = chunker.chunk_text(sample_text)

    # Display results
    print(
        f"\nFound {len(result.big_chunks)} big chunks and {len(result.small_chunks)} small chunks.\n"
    )

    print("Big chunks:")
    for i, chunk in enumerate(result.big_chunks):
        print(f"Chunk {i + 1}: {chunk[:50]}...\n")

    # Save results
    output_path = "chunking_results.json"
    result.save_to_file(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

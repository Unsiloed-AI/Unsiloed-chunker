import os
import Unsiloed
import json

# Set your OpenAI API key if you want to use semantic chunking
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def print_chunk_info(result, strategy_name):
    print(f"\n=== {strategy_name} Chunking Results ===")
    print(f"File type: {result['file_type']}")
    print(f"Total chunks: {result['total_chunks']}")
    print(f"Average chunk size: {result['avg_chunk_size']:.2f} characters")
    
    if result['chunks']:
        print(f"\nFirst chunk preview:")
        first_chunk = result['chunks'][0]
        print(f"Text: {first_chunk['text'][:100]}...")
        print(f"Metadata: {json.dumps(first_chunk['metadata'], indent=2)}")
    else:
        print("No chunks found")

# Example 1: Paragraph chunking (doesn't require OpenAI API key)
paragraph_result = Unsiloed.process_sync({
    "filePath": "README.md",
    "strategy": "paragraph"
})
print_chunk_info(paragraph_result, "Paragraph")

# Example 2: Fixed size chunking (doesn't require OpenAI API key)
fixed_result = Unsiloed.process_sync({
    "filePath": "README.md",
    "strategy": "fixed",
    "chunkSize": 500,
    "overlap": 50
})
print_chunk_info(fixed_result, "Fixed Size")

# Example 3: Heading chunking (doesn't require OpenAI API key)
heading_result = Unsiloed.process_sync({
    "filePath": "README.md",
    "strategy": "heading"
})
print_chunk_info(heading_result, "Heading")

# Example 4: Semantic chunking (requires OpenAI API key)
# Uncomment if you have set your OpenAI API key
'''
if os.environ.get("OPENAI_API_KEY"):
    semantic_result = Unsiloed.process_sync({
        "filePath": "README.md",
        "strategy": "semantic"
    })
    print_chunk_info(semantic_result, "Semantic")
else:
    print("\n=== Semantic Chunking ===")
    print("Skipped: OpenAI API key not set")
'''
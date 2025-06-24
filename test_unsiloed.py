import os
import Unsiloed

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Example using paragraph chunking (doesn't require OpenAI API key)
result = Unsiloed.process_sync({
    "filePath": "README.md",
    "strategy": "paragraph"
})

# Print the first chunk
if result["chunks"]:
    print(f"Total chunks: {result['total_chunks']}")
    print(f"First chunk: {result['chunks'][0]['text'][:100]}...")
else:
    print("No chunks found")
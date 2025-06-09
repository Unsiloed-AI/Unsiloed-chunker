import os
from Unsiloed.services.chunking import process_sync

# Example usage with a URL
result = process_sync({
    "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
    "credentials": {
        "apiKey": os.environ.get("OPENAI_API_KEY")
    },
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Print the number of chunks found
print(f"✅ Found {result['total_chunks']} chunks using strategy: {result['strategy']}")

# Print the first chunk's text
if result['chunks'] and len(result['chunks']) > 0:
    print("\n🧩 First chunk preview:")
    print(result['chunks'][0]['text'][:200] + "...")

"""
# Async example (uncomment to use if you implement `process()` async later)
import asyncio

async def main():
    result = await process({
        "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
        "credentials": {
            "apiKey": os.environ.get("OPENAI_API_KEY")
        }
    })
    print(f"Found {result['total_chunks']} chunks")

# asyncio.run(main())
"""

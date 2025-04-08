import os
import chunktopus

# Example usage with a URL
result = chunktopus.process_sync({
    "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
    "credentials": {
        "apiKey": os.environ.get("OPENAI_API_KEY")
    },
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

# Print the number of chunks found
print(f"Found {result['total_chunks']} chunks with strategy: {result['strategy']}")

# Print the first chunk's text
if result['chunks'] and len(result['chunks']) > 0:
    print("\nFirst chunk preview:")
    print(result['chunks'][0]['text'][:200] + "...")

"""
# Async example (uncomment to use)
import asyncio

async def main():
    result = await chunktopus.process({
        "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
        "credentials": {
            "apiKey": os.environ.get("OPENAI_API_KEY")
        }
    })
    print(f"Found {result['total_chunks']} chunks")

# asyncio.run(main())
""" 
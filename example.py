import os
import Unsiloed

# Test with JSON output (default)
json_result = Unsiloed.process_sync({
    "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
    "credentials": {
        "apiKey": os.environ.get("OPENAI_API_KEY")
    },
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100
})

print("\n=== JSON Output Test ===")
print(f"Found {json_result['total_chunks']} chunks with strategy: {json_result['strategy']}")
if json_result['chunks'] and len(json_result['chunks']) > 0:
    print("\nFirst chunk preview:")
    print(json_result['chunks'][0]['text'][:200] + "...")

# Test with Markdown output
markdown_result = Unsiloed.process_sync({
    "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
    "credentials": {
        "apiKey": os.environ.get("OPENAI_API_KEY")
    },
    "strategy": "semantic",
    "chunkSize": 1000,
    "overlap": 100,
    "outputFormat": "markdown"
})

print("\n=== Markdown Output Test ===")
# Save markdown output to a file
with open("output.md", "w") as f:
    f.write(markdown_result)
print("Markdown output saved to output.md")
print("\nFirst 500 characters of markdown output:")
print(markdown_result[:500] + "...")

"""
# Async example (uncomment to use)
import asyncio

async def main():
    result = await Unsiloed.process({
        "filePath": "https://omni-demo-data.s3.amazonaws.com/test/cs101.pdf",
        "credentials": {
            "apiKey": os.environ.get("OPENAI_API_KEY")
        }
    })
    print(f"Found {result['total_chunks']} chunks")

# asyncio.run(main())
""" 
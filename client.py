import requests
import json
import os
import argparse

def chunk_document(file_path, strategy="paragraph", chunk_size=1000, overlap=100, api_key=None, server_url="http://localhost:8000"):
    """Send a document to the Unsiloed API for chunking.
    
    Args:
        file_path: Path to the document file
        strategy: Chunking strategy (paragraph, fixed, heading, semantic)
        chunk_size: Size of chunks for fixed strategy
        overlap: Overlap size for fixed strategy
        api_key: OpenAI API key (required for semantic chunking)
        server_url: URL of the Unsiloed API server
        
    Returns:
        Dictionary with chunking results
    """
    # Prepare the request
    url = f"{server_url}/chunk"
    
    # Prepare form data
    form_data = {
        "strategy": strategy,
        "chunk_size": str(chunk_size),
        "overlap": str(overlap)
    }
    
    # Add API key if provided
    if api_key:
        form_data["api_key"] = api_key
    
    # Prepare file
    files = {
        "document": (os.path.basename(file_path), open(file_path, "rb"))
    }
    
    try:
        # Send the request
        response = requests.post(url, data=form_data, files=files)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    finally:
        # Close the file
        files["document"][1].close()

def print_chunk_info(result):
    """Print information about the chunking results."""
    if not result:
        return
    
    print(f"\n=== {result['strategy'].capitalize()} Chunking Results ===")
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Client for Unsiloed API")
    parser.add_argument("file_path", help="Path to the document file")
    parser.add_argument("--strategy", choices=["paragraph", "fixed", "heading", "semantic"], default="paragraph", help="Chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of chunks for fixed strategy")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap size for fixed strategy")
    parser.add_argument("--api-key", help="OpenAI API key (required for semantic chunking)")
    parser.add_argument("--server-url", default="http://localhost:8000", help="URL of the Unsiloed API server")
    parser.add_argument("--save", help="Save the results to a JSON file")
    
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.isfile(args.file_path):
        print(f"Error: File '{args.file_path}' not found")
        return
    
    # Check if semantic chunking is requested but no API key is provided
    if args.strategy == "semantic" and not args.api_key and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OpenAI API key is required for semantic chunking")
        print("Please provide it with --api-key or set the OPENAI_API_KEY environment variable")
        return
    
    # Use the API key from the environment if not provided
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    # Send the request
    result = chunk_document(
        args.file_path,
        args.strategy,
        args.chunk_size,
        args.overlap,
        api_key,
        args.server_url
    )
    
    # Print the results
    if result:
        print_chunk_info(result)
        
        # Save the results to a file if requested
        if args.save:
            with open(args.save, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.save}")

if __name__ == "__main__":
    main()
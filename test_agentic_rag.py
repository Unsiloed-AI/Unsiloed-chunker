#!/usr/bin/env python3
import os
import json
import argparse
import asyncio
import subprocess
import time
import requests
import signal
import sys
from dotenv import load_dotenv
from Unsiloed.services.retrieval import AgenticRetrieval

# Load environment variables from .env file (if present)
load_dotenv()

# Ensure OpenAI API key is set
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    api_key = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key

async def test_service_directly(pdf_path, delete_after=True):
    """Test the agentic RAG implementation directly using the service."""
    
    print(f"\n{'='*80}")
    print(f"PART 1: Testing Agentic RAG Service Directly")
    print(f"{'='*80}\n")
    
    # Initialize the retrieval service
    retrieval = AgenticRetrieval()
    document_id = None
    
    try:
        # Step 1: Index the PDF document
        print("\n--- Step 1: Indexing document ---")
        options = {
            "filePath": pdf_path,
            "fileType": "pdf",
            "strategy": "semantic",
            "chunkSize": 1000,
            "overlap": 100
        }
        
        index_result = await retrieval.index_document(options)
        document_id = index_result.get("document_id")
        
        print(f"Document ID: {document_id}")
        print(f"Indexed chunks: {index_result.get('indexed_chunks', 0)}")
        print(f"Status: {index_result.get('status')}")
        
        if index_result.get("status") != "success":
            print(f"Error: {index_result.get('error')}")
            return
        
        # Step 2: Test a simple query
        print("\n--- Step 2: Testing simple query ---")
        simple_query = "What is this document about?"
        
        print(f"Query: {simple_query}")
        simple_result = await retrieval.retrieve(simple_query)
        
        print(f"Query type: {simple_result.get('type')}")
        print(f"Found {len(simple_result.get('chunks', []))} chunks")
        
        # Print the first chunk's text (if available)
        if simple_result.get('chunks'):
            print("\nTop chunk:")
            print(f"Text: {simple_result['chunks'][0]['text'][:200]}...")
            print(f"Similarity: {simple_result['chunks'][0].get('similarity', 0):.4f}")
        
        # Display synthesized answer for simple queries
        if simple_result.get('answer'):
            print("\nSynthesized answer:")
            print(simple_result['answer'][:1000] + "..." if len(simple_result['answer']) > 1000 else simple_result['answer'])
        
        # Step 3: Test a multi-hop query
        print("\n--- Step 3: Testing multi-hop query ---")
        multi_hop_query = "What are the main concepts discussed and how are they related to each other?"
        
        print(f"Query: {multi_hop_query}")
        multi_hop_result = await retrieval.retrieve(multi_hop_query)
        
        print(f"Query type: {multi_hop_result.get('type')}")
        print(f"Number of sub-queries: {len(multi_hop_result.get('sub_queries', []))}")
        
        # Print sub-queries if present
        if multi_hop_result.get('sub_queries'):
            print("\nSub-queries:")
            for i, sub_query in enumerate(multi_hop_result['sub_queries']):
                print(f"{i+1}. {sub_query.get('query')}")
        
        # Print synthesized answer if available
        if multi_hop_result.get('answer'):
            print("\nSynthesized answer:")
            print(multi_hop_result['answer'][:1000] + "..." if len(multi_hop_result['answer']) > 1000 else multi_hop_result['answer'])
        
        # Step 4: Test a negation query
        print("\n--- Step 4: Testing negation query ---")
        negation_query = "What topics are covered in this document except for Final Thoughts?"
        
        print(f"Query: {negation_query}")
        negation_result = await retrieval.retrieve(negation_query)
        
        print(f"Query type: {negation_result.get('type')}")
        
        # Print synthesized answer if available
        if negation_result.get('answer'):
            print("\nSynthesized answer:")
            print(negation_result['answer'][:1000] + "..." if len(negation_result['answer']) > 1000 else negation_result['answer'])
        
        # Step 5: Test a more specific negation query
        print("\n--- Step 5: Testing specific negation query ---")
        specific_negation_query = "Show me all sections about career advice but not from the Final Thoughts section"
        
        print(f"Query: {specific_negation_query}")
        specific_negation_result = await retrieval.retrieve(specific_negation_query)
        
        print(f"Query type: {specific_negation_result.get('type')}")
        
        # Print sub-queries if present
        if specific_negation_result.get('sub_queries'):
            print("\nSub-queries:")
            for i, sub_query in enumerate(specific_negation_result['sub_queries']):
                print(f"{i+1}. {sub_query.get('query')}")
        
        # Print synthesized answer if available
        if specific_negation_result.get('answer'):
            print("\nSynthesized answer:")
            print(specific_negation_result['answer'][:1000] + "..." if len(specific_negation_result['answer']) > 1000 else specific_negation_result['answer'])
        
        print("\n--- Direct service tests completed successfully ---")
        
        return document_id
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return document_id

def test_api_endpoints(pdf_path):
    """Test the agentic RAG implementation via API endpoints."""
    
    print(f"\n{'='*80}")
    print(f"PART 2: Testing Agentic RAG via API Endpoints")
    print(f"{'='*80}\n")
    
    # Start the FastAPI server
    print("\n--- Starting Unsiloed app server ---")
    
    # Start the server process
    server_process = subprocess.Popen(
        ["uvicorn", "Unsiloed.main:app", "--port", "8765"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(5)  # Give the server some time to start
    
    base_url = "http://localhost:8765"
    document_id = None
    
    try:
        # Check if server is running
        health_response = requests.get(f"{base_url}/")
        if health_response.status_code != 200:
            print(f"Server not responding. Status code: {health_response.status_code}")
            return
            
        print("Server is running!")
        
        # Step 1: Index a document via API
        print("\n--- Step 1: Indexing document via API ---")
        with open(pdf_path, "rb") as pdf_file:
            files = {"document_file": (os.path.basename(pdf_path), pdf_file, "application/pdf")}
            data = {
                "strategy": "semantic",
                "chunk_size": "1000",
                "overlap": "100"
            }
            
            response = requests.post(
                f"{base_url}/retrieval/index",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                print(f"Error indexing document. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return
                
            index_result = response.json()
            document_id = index_result.get("document_id")
            
            print(f"Document ID: {document_id}")
            print(f"Indexed chunks: {index_result.get('indexed_chunks', 0)}")
            print(f"Status: {index_result.get('status')}")
        
        # Step 2: Test a simple query via API
        print("\n--- Step 2: Testing simple query via API ---")
        simple_query = "What is this document about?"
        
        print(f"Query: {simple_query}")
        query_data = {"query": simple_query, "top_k": 5}
        
        response = requests.post(
            f"{base_url}/retrieval/query",
            json=query_data
        )
        
        if response.status_code != 200:
            print(f"Error querying. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        simple_result = response.json()
        
        print(f"Query type: {simple_result.get('type')}")
        print(f"Found {len(simple_result.get('chunks', []))} chunks")
        
        # Print the first chunk's text (if available)
        if simple_result.get('chunks'):
            print("\nTop chunk:")
            print(f"Text: {simple_result['chunks'][0]['text'][:200]}...")
            print(f"Similarity: {simple_result['chunks'][0].get('similarity', 0):.4f}")
        
        # Display synthesized answer for simple queries
        if simple_result.get('answer'):
            print("\nSynthesized answer:")
            print(simple_result['answer'][:1000] + "..." if len(simple_result['answer']) > 1000 else simple_result['answer'])
        
        # Step 3: Test a multi-hop query via API
        print("\n--- Step 3: Testing multi-hop query via API ---")
        multi_hop_query = "What are the main concepts discussed and how are they related to each other?"
        
        print(f"Query: {multi_hop_query}")
        query_data = {"query": multi_hop_query, "top_k": 3}
        
        response = requests.post(
            f"{base_url}/retrieval/query",
            json=query_data
        )
        
        if response.status_code != 200:
            print(f"Error querying. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        multi_hop_result = response.json()
        
        print(f"Query type: {multi_hop_result.get('type')}")
        print(f"Number of sub-queries: {len(multi_hop_result.get('sub_queries', []))}")
        
        # Print synthesized answer if available
        if multi_hop_result.get('answer'):
            print("\nSynthesized answer:")
            print(multi_hop_result['answer'][:1000] + "..." if len(multi_hop_result['answer']) > 1000 else multi_hop_result['answer'])
        
        # Step 4: Test a negation query via API
        print("\n--- Step 4: Testing negation query via API ---")
        negation_query = "What topics are covered in this document except for Final Thoughts?"
        
        print(f"Query: {negation_query}")
        query_data = {"query": negation_query, "top_k": 5}
        
        response = requests.post(
            f"{base_url}/retrieval/query",
            json=query_data
        )
        
        if response.status_code != 200:
            print(f"Error querying. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        negation_result = response.json()
        
        print(f"Query type: {negation_result.get('type')}")
        
        # Print synthesized answer if available
        if negation_result.get('answer'):
            print("\nSynthesized answer:")
            print(negation_result['answer'][:1000] + "..." if len(negation_result['answer']) > 1000 else negation_result['answer'])
            
        # Step 5: Test a specific negation query via API
        print("\n--- Step 5: Testing specific negation query via API ---")
        specific_negation_query = "Show me all sections about career advice but not from the Final Thoughts section"
        
        print(f"Query: {specific_negation_query}")
        query_data = {"query": specific_negation_query, "top_k": 5}
        
        response = requests.post(
            f"{base_url}/retrieval/query",
            json=query_data
        )
        
        if response.status_code != 200:
            print(f"Error querying. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return
            
        specific_negation_result = response.json()
        
        print(f"Query type: {specific_negation_result.get('type')}")
        
        # Print synthesized answer if available
        if specific_negation_result.get('answer'):
            print("\nSynthesized answer:")
            print(specific_negation_result['answer'][:1000] + "..." if len(specific_negation_result['answer']) > 1000 else specific_negation_result['answer'])
        
        # Step 6: Delete the document via API
        if document_id:
            print(f"\n--- Step 6: Deleting document {document_id} via API ---")
            
            response = requests.delete(f"{base_url}/retrieval/document/{document_id}")
            
            if response.status_code != 200:
                print(f"Error deleting document. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return
                
            delete_result = response.json()
            
            print(f"Deletion status: {delete_result.get('status')}")
            print(f"Deleted chunks: {delete_result.get('deleted_chunks', 0)}")
        
        print("\n--- API endpoint tests completed successfully ---")
        
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the server. Make sure it's running.")
    except Exception as e:
        print(f"Error during API testing: {str(e)}")
    finally:
        # Terminate the server process
        print("\n--- Shutting down server ---")
        try:
            os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
            server_process.wait(timeout=5)
            print("Server shutdown complete")
        except Exception as e:
            print(f"Error shutting down server: {str(e)}")
            # Force kill if necessary
            try:
                os.killpg(os.getpgid(server_process.pid), signal.SIGKILL)
            except:
                pass

async def main(pdf_path, keep=False, test_api=True):
    """Main test function that runs both direct service and API endpoint tests."""
    
    # Part 1: Test the service directly
    document_id = await test_service_directly(pdf_path, delete_after=not keep)
    
    # Part 2: Test the API endpoints
    if test_api:
        test_api_endpoints(pdf_path)
    
    # Clean up if keep is False and we have a document_id
    if not keep and document_id:
        print(f"\n--- Cleaning up: Deleting document {document_id} ---")
        try:
            retrieval = AgenticRetrieval()
            delete_result = await retrieval.delete_document(document_id)
            print(f"Deletion status: {delete_result.get('status')}")
            print(f"Deleted chunks: {delete_result.get('deleted_chunks', 0)}")
        except Exception as e:
            print(f"Error deleting document: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test agentic RAG implementation on a PDF file")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--keep", action="store_true", help="Keep the document in the database after testing")
    parser.add_argument("--no-api", action="store_true", help="Skip testing API endpoints")
    
    args = parser.parse_args()
    
    # Run the test
    asyncio.run(main(args.pdf_path, args.keep, not args.no_api)) 
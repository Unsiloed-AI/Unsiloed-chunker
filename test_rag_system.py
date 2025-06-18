import os
import sys
import tempfile
from typing import Dict, Any, List

# Import the RAG system components
from rag_system import InMemoryVectorStore, create_embedding, Agent
import Unsiloed

def test_vector_store():
    """Test the in-memory vector store"""
    print("\n=== Testing Vector Store ===")
    
    # Initialize vector store
    vector_store = InMemoryVectorStore()
    
    # Add some test documents
    vector_store.add_document(
        "doc1", 
        "This is a test document about artificial intelligence.",
        [0.1, 0.2, 0.3],  # Simplified test embedding
        {"source": "test.md"}
    )
    
    vector_store.add_document(
        "doc2", 
        "Python is a programming language used for AI and data science.",
        [0.2, 0.3, 0.4],  # Simplified test embedding
        {"source": "test.md"}
    )
    
    # Test search functionality
    results = vector_store.search([0.15, 0.25, 0.35], top_k=2)
    
    # Verify results
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["id"] in ["doc1", "doc2"], f"Expected doc1 or doc2, got {results[0]['id']}"
    
    print("✅ Vector store test passed")
    return vector_store

def test_document_processing():
    """Test document processing with Unsiloed"""
    print("\n=== Testing Document Processing ===")
    
    # Create a temporary test document (using .md extension since Unsiloed supports Markdown)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
        temp_file.write("""# Test Document
        
        This is a test document for the RAG system.
        
        ## Section 1
        
        The RAG (Retrieval-Augmented Generation) approach combines retrieval systems with generative models.
        
        ## Section 2
        
        This approach helps provide more accurate and contextually relevant responses.
        """)
        temp_file_path = temp_file.name
    
    try:
        # Process the document using Unsiloed
        result = Unsiloed.process_sync({
            "filePath": temp_file_path,
            "strategy": "paragraph",  # Use paragraph strategy as it doesn't require OpenAI API key
            "chunkSize": 1000,
            "overlap": 100
        })
        
        # Verify results
        assert "chunks" in result, "Expected 'chunks' in result"
        assert len(result["chunks"]) > 0, "Expected at least one chunk"
        
        print(f"✅ Document processing test passed - Created {len(result['chunks'])} chunks")
        print(f"First chunk: {result['chunks'][0]['text'][:100]}...")
        
        return result
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def test_embedding_creation():
    """Test embedding creation"""
    print("\n=== Testing Embedding Creation ===")
    
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not found, using random embeddings for testing")
    
    # Create an embedding
    text = "This is a test document for embedding creation."
    embedding = create_embedding(text)
    
    # Verify embedding
    assert isinstance(embedding, list), f"Expected list, got {type(embedding)}"
    assert len(embedding) > 0, "Expected non-empty embedding"
    
    print(f"✅ Embedding creation test passed - Created embedding of length {len(embedding)}")
    return embedding

def test_agent(vector_store):
    """Test the agent's query processing"""
    print("\n=== Testing Agent ===")
    
    # Initialize agent
    agent = Agent(vector_store)
    
    # Add test documents with actual embeddings
    doc1_text = "RAG systems combine retrieval with generative AI to provide accurate responses."
    doc1_embedding = create_embedding(doc1_text)
    vector_store.add_document("doc1", doc1_text, doc1_embedding, {"source": "test.md"})
    
    doc2_text = "Vector databases store embeddings for efficient similarity search."
    doc2_embedding = create_embedding(doc2_text)
    vector_store.add_document("doc2", doc2_text, doc2_embedding, {"source": "test.md"})
    
    # Process a query
    query = "What is a RAG system?"
    
    # Check if OpenAI API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ OpenAI API key not found, agent will return a placeholder response")
    
    # Process the query
    import asyncio
    result = asyncio.run(agent.process_query(query))
    
    # Verify result
    assert "answer" in result, "Expected 'answer' in result"
    assert "sources" in result, "Expected 'sources' in result"
    
    print(f"✅ Agent test passed")
    print(f"Query: {query}")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Sources: {len(result['sources'])}")
    
    return result

def run_all_tests():
    """Run all tests"""
    print("🧪 Running RAG System Tests")
    
    try:
        # Test vector store
        vector_store = test_vector_store()
        
        # Test document processing
        test_document_processing()
        
        # Test embedding creation
        test_embedding_creation()
        
        # Test agent
        test_agent(vector_store)
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
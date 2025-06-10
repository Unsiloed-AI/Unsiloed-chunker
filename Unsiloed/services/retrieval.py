import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from Unsiloed.utils.vector_db import SQLiteVectorDB
from Unsiloed.utils.embeddings import generate_embeddings, generate_query_embedding
from Unsiloed.utils.agent import analyze_query, decompose_query, synthesize_results, QueryType
from Unsiloed.services.chunking import process_document_chunking

logger = logging.getLogger(__name__)

class AgenticRetrieval:
    """Agentic retrieval service for complex queries."""
    
    def __init__(self, db_path: str = "vector_store.db"):
        """Initialize the agentic retrieval service.
        
        Args:
            db_path: Path to the SQLite vector database
        """
        self.vector_db = SQLiteVectorDB(db_path)
        
    async def index_document(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process and index a document for retrieval.
        
        Args:
            options: Dictionary with processing options
                - filePath: Path to the document
                - documentId: Optional ID for the document (generated if not provided)
                - strategy: Chunking strategy
                - other options passed to process_document_chunking
                
        Returns:
            Dictionary with indexing results
        """
        # Generate document ID if not provided
        document_id = options.get("documentId", str(uuid.uuid4()))
        options["documentId"] = document_id
        
        try:
            # Process document with existing chunking service
            chunking_result = process_document_chunking(
                file_path=options["filePath"],
                file_type=options.get("fileType", "pdf"),
                strategy=options.get("strategy", "semantic"),
                chunk_size=options.get("chunkSize", 1000),
                overlap=options.get("overlap", 100)
            )
            
            # Extract chunks
            chunks = chunking_result.get("chunks", [])
            if not chunks:
                logger.warning(f"No chunks were generated for document {document_id}")
                return {
                    "document_id": document_id,
                    "status": "error",
                    "error": "No chunks were generated",
                    "indexed_chunks": 0
                }
                
            # Generate embeddings for all chunks
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = generate_embeddings(chunk_texts)
            
            # Store chunks and embeddings in vector database
            self.vector_db.add_chunks(chunks, document_id, embeddings)
            
            return {
                "document_id": document_id,
                "status": "success",
                "indexed_chunks": len(chunks),
                "chunking_strategy": options.get("strategy", "semantic")
            }
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e),
                "indexed_chunks": 0
            }
    
    async def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform agentic retrieval for a query.
        
        Args:
            query: The user's query
            top_k: Number of top results to return per sub-query
            
        Returns:
            Dictionary with retrieval results
        """
        try:
            # Analyze the query
            analysis = analyze_query(query)
            query_type = analysis.get("type", QueryType.UNKNOWN)
            
            logger.info(f"Query '{query}' analyzed as type: {query_type}")
            
            # For simple queries, perform direct retrieval
            if query_type == QueryType.SIMPLE:
                # Generate embedding for the query
                query_embedding = generate_query_embedding(query)
                
                # Search the vector database
                results = self.vector_db.search(query_embedding, top_k=top_k)
                
                return {
                    "query": query,
                    "type": query_type,
                    "chunks": results,
                    "sub_queries": []
                }
                
            # For complex queries, use agentic approach
            else:
                # Decompose into sub-queries
                sub_queries = decompose_query(query)
                sub_results = []
                
                # Process each sub-query
                for sub_query in sub_queries:
                    sub_query_text = sub_query["query"]
                    
                    # Generate embedding for sub-query
                    sub_embedding = generate_query_embedding(sub_query_text)
                    
                    # Search for this sub-query
                    sub_chunks = self.vector_db.search(sub_embedding, top_k=top_k)
                    
                    # Store sub-query results
                    sub_result = {
                        "query": sub_query_text,
                        "step": sub_query.get("step", 0),
                        "chunks": sub_chunks,
                        "metadata": sub_query
                    }
                    sub_results.append(sub_result)
                
                # For negation queries, filter out negated items
                if query_type == QueryType.NEGATION:
                    negated_entity = analysis.get("negated_entity", "")
                    if negated_entity:
                        for result in sub_results:
                            # Filter chunks that contain the negated entity
                            result["chunks"] = [
                                chunk for chunk in result["chunks"] 
                                if negated_entity.lower() not in chunk["text"].lower()
                            ]
                
                # Synthesize results from all sub-queries
                synthesis = synthesize_results(sub_results, query)
                
                return {
                    "query": query,
                    "type": query_type,
                    "analysis": analysis,
                    "sub_queries": sub_results,
                    "answer": synthesis.get("synthesized_answer", ""),
                    "chunks": [chunk for result in sub_results for chunk in result["chunks"]]
                }
                
        except Exception as e:
            logger.error(f"Error in agentic retrieval: {str(e)}")
            return {
                "query": query,
                "type": "error",
                "error": str(e),
                "chunks": []
            }
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the vector database.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            deleted_count = self.vector_db.delete_document(document_id)
            
            return {
                "document_id": document_id,
                "status": "success",
                "deleted_chunks": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e)
            } 
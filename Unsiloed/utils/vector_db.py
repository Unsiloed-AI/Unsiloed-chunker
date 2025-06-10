import sqlite3
import json
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SQLiteVectorDB:
    """SQLite-based vector database for storing and retrieving document chunks."""
    
    def __init__(self, db_path: str = "vector_store.db"):
        """Initialize the SQLite vector database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
        
    def _initialize_db(self):
        """Create the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create chunks table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            document_id TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create embeddings table with extension for vector similarity
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER,
            embedding BLOB NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks (id) ON DELETE CASCADE
        )
        ''')
        
        # Create index on document_id for faster retrieval
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON chunks (document_id)')
        
        conn.commit()
        conn.close()
        
    def add_chunks(self, chunks: List[Dict[str, Any]], document_id: str, embeddings: List[np.ndarray]):
        """Add chunks and their embeddings to the database.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            document_id: ID of the document these chunks belong to
            embeddings: List of embedding vectors corresponding to chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for i, chunk in enumerate(chunks):
                # Insert chunk
                cursor.execute(
                    'INSERT INTO chunks (text, document_id, metadata) VALUES (?, ?, ?)',
                    (
                        chunk['text'], 
                        document_id, 
                        json.dumps(chunk.get('metadata', {}))
                    )
                )
                chunk_id = cursor.lastrowid
                
                # Insert embedding
                embedding_bytes = embeddings[i].tobytes()
                cursor.execute(
                    'INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)',
                    (chunk_id, embedding_bytes)
                )
            
            conn.commit()
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding chunks to vector database: {str(e)}")
            raise
        finally:
            conn.close()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for most similar chunks using cosine similarity.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of top results to return
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        try:
            # Get all embeddings and calculate similarity
            cursor.execute('SELECT chunk_id, embedding FROM embeddings')
            rows = cursor.fetchall()
            
            similarities = []
            for chunk_id, embedding_bytes in rows:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                
                # Normalize vectors for cosine similarity
                norm_query = query_embedding / np.linalg.norm(query_embedding)
                norm_embedding = embedding / np.linalg.norm(embedding)
                
                # Calculate cosine similarity
                similarity = np.dot(norm_query, norm_embedding)
                similarities.append((chunk_id, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:top_k]
            
            # Get chunk details for top results
            for chunk_id, similarity in top_similarities:
                cursor.execute(
                    'SELECT text, document_id, metadata FROM chunks WHERE id = ?', 
                    (chunk_id,)
                )
                text, document_id, metadata_json = cursor.fetchone()
                metadata = json.loads(metadata_json)
                
                results.append({
                    'text': text,
                    'document_id': document_id,
                    'metadata': metadata,
                    'similarity': float(similarity)
                })
                
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            raise
        finally:
            conn.close()
            
        return results
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Number of chunks deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get chunk IDs to delete
            cursor.execute('SELECT id FROM chunks WHERE document_id = ?', (document_id,))
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete embeddings first (foreign key constraint)
            for chunk_id in chunk_ids:
                cursor.execute('DELETE FROM embeddings WHERE chunk_id = ?', (chunk_id,))
            
            # Delete chunks
            cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error deleting document: {str(e)}")
            raise
        finally:
            conn.close() 
import numpy as np
from typing import List, Dict, Any, Union
import logging
from Unsiloed.utils.openai import get_openai_client

logger = logging.getLogger(__name__)

def generate_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings for a list of texts using OpenAI API.
    
    Args:
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
        
    try:
        client = get_openai_client()
        if not client:
            logger.error("Failed to initialize OpenAI client")
            raise ValueError("OpenAI client initialization failed")
            
        # Process texts in batches to avoid API limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Call OpenAI embeddings API
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts,
                encoding_format="float"
            )
            
            # Extract embedding vectors from response
            batch_embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
            
        return all_embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for a single query string.
    
    Args:
        query: The query text
        
    Returns:
        Embedding vector for the query
    """
    embeddings = generate_embeddings([query])
    if not embeddings:
        raise ValueError("Failed to generate query embedding")
    return embeddings[0] 
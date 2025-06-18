import os
import json
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import Unsiloed
import tempfile
import shutil
from pydantic import BaseModel, Field
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")

# Simple in-memory vector store
class InMemoryVectorStore:
    def __init__(self):
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
    
    def add_document(self, doc_id: str, text: str, embedding: List[float], metadata: Dict[str, Any]):
        self.documents[doc_id] = text
        self.embeddings[doc_id] = embedding
        self.metadata[doc_id] = metadata
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        
        # Calculate cosine similarity
        similarities = {}
        for doc_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities[doc_id] = similarity
        
        # Sort by similarity (descending)
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_results:
            results.append({
                "id": doc_id,
                "text": self.documents[doc_id],
                "metadata": self.metadata[doc_id],
                "score": float(score)
            })
        
        return results
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Initialize vector store
vector_store = InMemoryVectorStore()

# Create embeddings using OpenAI
def create_embedding(text: str) -> List[float]:
    if not openai_api_key:
        # Return random embedding for testing if no API key
        return list(np.random.rand(1536))
    
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Agent for handling queries
class Agent:
    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store
    
    async def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # Create embedding for the query
        query_embedding = create_embedding(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": []
            }
        
        # Construct prompt with retrieved documents
        context = "\n\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        # Generate response using OpenAI
        if openai_api_key:
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided documents. If the answer is not in the documents, say 'I don't have enough information to answer that question.'"},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            answer = response.choices[0].message.content
        else:
            answer = "API key not provided. Please set the OPENAI_API_KEY environment variable."
        
        return {
            "answer": answer,
            "sources": retrieved_docs
        }

    def process_query_sync(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Synchronous version of process_query"""
        import asyncio
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.process_query(query, top_k=top_k))
        loop.close()
        return result

# Initialize agent
agent = Agent(vector_store)
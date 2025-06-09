"""
Core implementation of the agentic RAG retrieval system.
"""
import os
import shutil
import uuid
import tempfile
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.schema import Document
from langchain.agents import AgentType, initialize_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class AgenticRAG:
    """
    Agentic RAG retrieval system for multi-hop queries and negation queries.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        temperature: float = 0,
        embedding_model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the AgenticRAG system.
        
        Args:
            model_name: The name of the OpenAI model to use
            temperature: The temperature parameter for the LLM
            embedding_model: The name of the embedding model to use
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            persist_directory: Directory to persist vector store (if None, a temp dir will be used)
        """
        # Use provided API key or fall back to environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=self.api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=self.api_key
        )
        
        # Set persistence directory
        self.persist_directory = persist_directory or tempfile.mkdtemp(prefix="agentic_rag_")
        self.use_temp_dir = persist_directory is None  # Flag to track if we're using a temp dir
        
        # Vector store will be initialized when documents are added
        self.vector_store = None
        self.documents = []
        
        # Initialize the agent with tools
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the agent with necessary tools."""
        # Define the tools
        tools = [
            Tool(
                name="Search",
                func=self._search_documents,
                description="Useful for searching relevant information in the documents. Input should be a search query."
            ),
            Tool(
                name="Decompose",
                func=self._decompose_query,
                description="Useful for breaking down complex queries into simpler sub-queries. Input should be a complex query."
            ),
            Tool(
                name="FilterNegation",
                func=self._filter_negation,
                description="Useful for filtering out documents that contain negated concepts. Input should be formatted as 'QUERY: <query with negation>' or 'QUERY: <query with negation>\\nDOCUMENTS: <documents>'"
            )
        ]
        
        # System message content for the ReAct agent
        system_message = """
        You are an advanced retrieval agent that helps find information in documents.
        You can handle multi-hop queries by breaking them down into steps, and you can
        handle negation queries by filtering out irrelevant information.
        
        Think through this step by step:
        1. Is this a simple query, a multi-hop query, or a negation query?
        2. For multi-hop queries, break down the question into sub-questions.
        3. For negation queries, identify what should be excluded.
        4. Search for relevant information using the appropriate tools.
        """
        
        # Set up the agent with tools - use ZERO_SHOT_REACT_DESCRIPTION instead
        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document dictionaries with 'text' and optional metadata
        """
        # Convert to LangChain Document format
        langchain_docs = []
        for i, doc in enumerate(documents):
            if isinstance(doc, dict) and 'text' in doc:
                metadata = {k: v for k, v in doc.items() if k != 'text'}
                if 'id' not in metadata:
                    metadata['id'] = str(i)
                langchain_docs.append(Document(page_content=doc['text'], metadata=metadata))
            else:
                raise ValueError(f"Document at index {i} does not have 'text' field")
        
        # Store documents
        self.documents.extend(langchain_docs)
        
        # Create or update vector store with persistence
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                langchain_docs,
                self.embeddings
            )
            # Save to disk
            self.vector_store.save_local(self.persist_directory)
        else:
            # Add new documents to existing vector store
            self.vector_store.add_documents(langchain_docs)
            # Update persisted version
            self.vector_store.save_local(self.persist_directory)
        
        print(f"Vector store saved to {self.persist_directory}")
    
    def _search_documents(self, query: str, k: int = 5) -> str:
        """
        Search documents using vector similarity.
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            String representation of search results
        """
        if self.vector_store is None:
            return "No documents have been added to the system yet."
        
        docs = self.vector_store.similarity_search(query, k=k)
        
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Document {i+1}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    
    def _decompose_query(self, query: str) -> str:
        """
        Decompose a complex query into simpler sub-queries.
        
        Args:
            query: The complex query to decompose
            
        Returns:
            String representation of sub-queries
        """
        # Create a modern runnable chain using LCEL (LangChain Expression Language)
        prompt = ChatPromptTemplate.from_template("""
        You are an expert at breaking down complex questions into simpler sub-questions.
        
        Complex Question: {query}
        
        Break down this question into 2-4 simpler sub-questions that would help answer the original question.
        For each sub-question, explain why it's necessary.
        
        Output format:
        Sub-question 1: [question]
        Reason: [reason]
        
        Sub-question 2: [question]
        Reason: [reason]
        
        ...and so on.
        """)
        
        # Create the chain using the modern pattern
        chain = prompt | self.llm | StrOutputParser()
        
        # Run the chain
        response = chain.invoke({"query": query})
        
        return response
    
    def _filter_negation(self, input_str: str) -> str:
        """
        Filter documents based on negation criteria.
        
        Args:
            input_str: String containing the negation query and optionally documents
                       Format should be: "QUERY: <query with negation>\nDOCUMENTS: <documents>"
                       If no documents are provided, all available documents will be searched.
            
        Returns:
            String representation of filtered documents
        """
        # Parse input string
        parts = input_str.split("DOCUMENTS:", 1)
        query = parts[0].replace("QUERY:", "").strip()
        
        # Get documents from input or search for documents
        if len(parts) > 1 and parts[1].strip():
            documents = parts[1].strip()
        else:
            # If no documents provided, search for all relevant documents
            documents = self._search_documents("*", k=10)
        
        # Create a modern runnable chain using LCEL
        prompt = ChatPromptTemplate.from_template("""
        You are an expert at handling negation in search queries.
        
        Query with negation: {query}
        
        Documents to filter:
        {documents}
        
        Identify the negation criteria in the query and filter out documents that match what should be excluded.
        For each document, explain whether it should be included or excluded and why.
        
        Output format:
        Document 1: [Include/Exclude]
        Reason: [reason]
        
        Document 2: [Include/Exclude]
        Reason: [reason]
        
        ...and so on.
        """)
        
        # Create the chain using the modern pattern
        chain = prompt | self.llm | StrOutputParser()
        
        # Run the chain
        response = chain.invoke({"query": query, "documents": documents})
        
        return response
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the agentic RAG system.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing the response and any additional information
        """
        if self.vector_store is None:
            return {
                "answer": "No documents have been added to the system yet.",
                "sources": [],
                "reasoning": "Cannot perform retrieval without documents."
            }
        
        # Run the agent - ZERO_SHOT_REACT_DESCRIPTION doesn't use chat_history
        result = self.agent_executor.invoke({"input": query})
        
        # Extract the answer from the result
        answer = result.get("output", "")
        
        # Extract the thought process from the intermediate steps if available
        reasoning = ""
        if "intermediate_steps" in result:
            steps = result["intermediate_steps"]
            reasoning = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "raw_result": result
        }
    
    def cleanup(self):
        """
        Clean up resources, including deleting the persisted vector store files.
        """
        if self.use_temp_dir and os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print(f"Cleaned up temporary vector store at {self.persist_directory}")
                self.persist_directory = None
            except Exception as e:
                print(f"Error cleaning up vector store: {str(e)}")
    
    def __del__(self):
        """
        Destructor to clean up resources automatically when the object is deleted.
        """
        self.cleanup() 
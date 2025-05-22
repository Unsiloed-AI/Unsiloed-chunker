import os
import hashlib
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List
from openai import OpenAI, RateLimitError
from functools import lru_cache
from dotenv import load_dotenv
import cachetools  # More efficient caching
from cachetools import TTLCache, LRUCache  # Time-based and LRU caching

load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIClientSingleton:
    """
    Singleton class for OpenAI client with caching functionality.
    
    This class:
    1. Ensures only one client instance is created
    2. Provides advanced caching for API requests to reduce latency and costs
    3. Handles connection errors and retries with exponential backoff
    4. Implements batch processing for multiple requests
    """
    _instance = None
    _lock = threading.Lock()
    # Use TTLCache for better memory management with time-based expiration
    _cache = TTLCache(maxsize=1000, ttl=3600)  # Cache entries expire after 1 hour
    _request_cache = LRUCache(maxsize=500)  # For short-term request deduplication
    
    # Rate limiting and retry parameters
    _max_retries = 5
    _retry_delay = 1  # Initial delay in seconds
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(OpenAIClientSingleton, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the client"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY environment variable is not set")
                raise ValueError("OPENAI_API_KEY environment variable is not set")
                
            self.client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
            logger.debug("OpenAI client singleton initialized successfully")
            
            # No need to test the client by listing models - this adds unnecessary latency
            # We'll use the client directly and handle any errors at usage time
            
            # Initialize cache
            self._cache = {}
            
        except Exception as e:
            logger.error(f"Error initializing OpenAI client singleton: {str(e)}")
            raise
    
    def get_client(self) -> OpenAI:
        """Get the OpenAI client instance"""
        return self.client
    
    def generate_cache_key(self, model: str, messages: list, **kwargs) -> str:
        """
        Generate a cache key from the request parameters.
        
        Args:
            model: The model name
            messages: The messages for the completion
            **kwargs: Additional parameters for the completion
            
        Returns:
            A string hash to use as a cache key
        """
        # Create a dictionary of all parameters
        cache_dict = {
            "model": model,
            "messages": messages,
        }
        
        # Add other parameters that affect the output
        for key in ["temperature", "max_tokens", "response_format"]:
            if key in kwargs:
                cache_dict[key] = kwargs[key]
        
        # Convert to a string and hash
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def chat_completion_with_cache(self, model: str, messages: list, use_cache: bool = True, **kwargs) -> Any:
        """
        Get a chat completion with caching.
        
        Args:
            model: The model name
            messages: The messages for the completion
            use_cache: Whether to use the cache (default: True)
            **kwargs: Additional parameters for the completion
            
        Returns:
            The completion response
        """
        if use_cache:
            cache_key = self.generate_cache_key(model, messages, **kwargs)
            
            # Check if in cache
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key[:10]}...")
                return self._cache[cache_key]
        
        # If not in cache or cache disabled, make the API call
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            # Cache the result if caching is enabled
            if use_cache:
                self._cache[cache_key] = response
                logger.debug(f"Cached response for key: {cache_key[:10]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the number of items in the cache"""
        return len(self._cache)
    
    def batch_process_completion(self, messages_list: List[List], model: str = "gpt-4o", use_cache: bool = True, **kwargs) -> List[Any]:
        """
        Process multiple completion requests in a batch to minimize API call overhead.
        
        Args:
            messages_list: List of message arrays, each one for a separate completion request
            model: OpenAI model to use
            use_cache: Whether to use cache for requests
            **kwargs: Additional parameters to pass to OpenAI API
            
        Returns:
            List of completion results corresponding to each input message array
        """
        if not messages_list:
            return []
            
        results = []
        cache_hits = 0
        batch_sizes = []
        
        # First check cache for each request
        if use_cache:
            cached_results = []
            uncached_indices = []
            uncached_messages = []
            
            for i, messages in enumerate(messages_list):
                # Generate a cache key for this specific request
                cache_key = self._generate_cache_key(model, messages, kwargs)
                
                if cache_key in self._cache:
                    cached_results.append((i, self._cache[cache_key]))
                    cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_messages.append(messages)
                    
            # Sort cached results by index
            cached_results.sort(key=lambda x: x[0])
            
            if len(uncached_messages) == 0:
                # All requests were cached, return them in original order
                return [result for _, result in cached_results]
                
            # Process non-cached requests in batches to optimize API usage
            processed_results = self._batch_api_calls(uncached_messages, model, **kwargs)
            
            # Cache the new results
            for i, result in zip(uncached_indices, processed_results):
                cache_key = self._generate_cache_key(model, messages_list[i], kwargs)
                self._cache[cache_key] = result
                
            # Merge cached and new results maintaining original order
            all_results = [None] * len(messages_list)
            for i, result in cached_results:
                all_results[i] = result
                
            for i, result in zip(uncached_indices, processed_results):
                all_results[i] = result
                
            return all_results
        else:
            # Skip caching, process all requests
            return self._batch_api_calls(messages_list, model, **kwargs)
        
    def _batch_api_calls(self, messages_list: List[List], model: str, **kwargs) -> List[Any]:
        """
        Make batched API calls to OpenAI with retry logic
        """
        results = []
        
        for messages in messages_list:
            retry_count = 0
            delay = self._retry_delay
            
            while retry_count <= self._max_retries:
                try:
                    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    results.append(response)
                    break
                except RateLimitError:
                    if retry_count == self._max_retries:
                        raise
                    logger.warning(f"Rate limit exceeded, retrying in {delay} seconds...")
                    time.sleep(delay)
                    # Exponential backoff
                    delay *= 2
                    retry_count += 1
                except Exception as e:
                    if retry_count == self._max_retries:
                        raise
                    logger.warning(f"API call failed: {str(e)}, retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                    retry_count += 1
        
        return results
        
    def _generate_cache_key(self, model: str, messages: list, kwargs: Dict) -> str:
        """Generate a deterministic cache key for a request"""
        # Create a dict with all parameters that affect the response
        key_dict = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k != "stream"}  # Exclude stream parameter
        }
        # Convert to JSON string and hash
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()
        

# Global function to get the client
def get_optimized_openai_client() -> OpenAI:
    """Get the optimized OpenAI client with caching"""
    return OpenAIClientSingleton().get_client()


# Global function for chat completion with caching
def chat_completion_with_cache(model: str, messages: list, use_cache: bool = True, **kwargs) -> Any:
    """
    Get a chat completion with caching.
    
    Args:
        model: The model name
        messages: The messages for the completion
        use_cache: Whether to use the cache (default: True)
        **kwargs: Additional parameters for the completion
        
    Returns:
        The completion response
    """
    return OpenAIClientSingleton().chat_completion_with_cache(model, messages, use_cache, **kwargs)

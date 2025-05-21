"""
Document processing cache for improved performance.

This module provides advanced caching functionality for document processing
to avoid redundant processing of the same documents with multiple cache strategies.
"""
import os
import hashlib
import json
import logging
import time
import traceback
from typing import Dict, Any, Optional, Tuple, Union, BinaryIO
from functools import lru_cache
import threading
import cachetools
from cachetools import TTLCache, LRUCache, cached

# Import custom exceptions - wrapped in try/except to avoid circular imports
try:
    from Unsiloed.utils.exceptions import CacheError
    CUSTOM_EXCEPTIONS_AVAILABLE = True
except ImportError:
    CUSTOM_EXCEPTIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentCache:
    """
    Advanced cache for document processing results.
    
    This class:
    1. Caches document processing results using file content hashes as keys
    2. Implements TTL (time-to-live) and LRU (least recently used) caching strategies
    3. Avoids redundant processing of the same document
    4. Improves performance for repeated operations
    5. Manages memory efficiently through automatic cache expiration
    """
    _instance = None
    _lock = threading.Lock()
    
    # Use multiple cache types for different use cases
    # TTLCache: For frequently accessed documents with time-based expiration
    # LRUCache: For large document results that should be evicted when memory pressure increases
    _ttl_cache: TTLCache  # Time-based expiration cache
    _lru_cache: LRUCache  # Least Recently Used cache for memory-sensitive operations
    _metadata: Dict[str, Dict[str, Any]] = {}  # Store metadata like access counts, timestamps
    
    # Cache configuration
    _ttl_seconds = 3600  # Default TTL: 1 hour
    _ttl_max_size = 100  # Maximum number of TTL cache entries
    _lru_max_size = 50   # Maximum number of LRU cache entries (typically larger documents)
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DocumentCache, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the cache with multiple caching strategies"""
        self._ttl_cache = TTLCache(maxsize=self._ttl_max_size, ttl=self._ttl_seconds)
        self._lru_cache = LRUCache(maxsize=self._lru_max_size)
        self._metadata = {}
        logger.debug("Document cache initialized with TTL and LRU caching strategies")
    
    def get_hash(self, file_path: str) -> str:
        """
        Get a hash of the file's content to use as cache key.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash string representing the file's content
        """
        try:
            # If we have a valid modification timestamp, use it to detect file changes
            file_stat = os.stat(file_path)
            file_size = file_stat.st_size
            mod_time = file_stat.st_mtime
            
            # For small files, use content hash; for large files, use size+mtime for performance
            if file_size < 50 * 1024 * 1024:  # 50MB threshold
                hasher = hashlib.md5()
                with open(file_path, 'rb') as file:
                    # Read in chunks to handle large files efficiently
                    chunk = file.read(65536)
                    while chunk:
                        hasher.update(chunk)
                        chunk = file.read(65536)
                return hasher.hexdigest()
            else:
                # For very large files, use file metadata as a faster approximation
                meta_str = f"{file_path}_{file_size}_{mod_time}"
                return hashlib.md5(meta_str.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate hash for {file_path}: {str(e)}")
            # Fall back to using the file path if we can't read the file
            return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    def _select_cache_for_result(self, result: Dict[str, Any]) -> str:
        """
        Determine which cache to use based on result characteristics.
        
        Args:
            result: The processing result to analyze
            
        Returns:
            Cache type to use: 'ttl' or 'lru'
        """
        # Estimate result size - we'll use LRU for larger results
        try:
            # Check if it contains text chunks (which can be large)
            if 'chunks' in result and len(result.get('chunks', [])) > 50:
                return 'lru'
            
            # Rough estimate of size by JSON serialization length
            size = len(json.dumps(result))
            if size > 100000:  # If larger than ~100KB
                return 'lru'
        except Exception:
            pass  # Default to TTL cache if we can't determine size
            
        return 'ttl'
    
    def get(self, file_path: Union[str, BinaryIO], strategy: str) -> Optional[Dict[str, Any]]:
        """
        Get cached processing result for a file and strategy.
        
        Args:
            file_path: Path to the file or file-like object
            strategy: Chunking strategy used
            
        Returns:
            Cached result or None if not in cache
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        # Generate a unique key combining file hash and strategy
        file_id = str(file_path) if isinstance(file_path, str) else "file_object"
        
        try:
            file_hash = self.get_hash(file_path)
            cache_key = f"{file_hash}_{strategy}"
            
            # Check TTL cache first (for frequently accessed smaller docs)
            result = self._ttl_cache.get(cache_key)
            if result is not None:
                logger.debug(f"TTL cache hit for {file_id}")
                self._update_metadata(cache_key, 'ttl')
                return result
                
            # Then check LRU cache (for larger docs)
            result = self._lru_cache.get(cache_key)
            if result is not None:
                logger.debug(f"LRU cache hit for {file_id}")
                self._update_metadata(cache_key, 'lru')
                return result
                
            # No cache hit
            logger.debug(f"Cache miss for {file_id} with strategy {strategy}")
            return None
            
        except Exception as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
            error_details = {
                "error_type": type(e).__name__,
                "strategy": strategy,
                "traceback": traceback.format_exc()
            }
            
            # Use custom exception if available
            if CUSTOM_EXCEPTIONS_AVAILABLE:
                raise CacheError(
                    f"Failed to retrieve from cache: {str(e)}", 
                    operation="get",
                    file_path=file_id if isinstance(file_path, str) else None,
                    details=error_details
                ) from e
            else:
                # Otherwise use a standard exception
                raise RuntimeError(f"Cache retrieval error: {str(e)}, details: {error_details}") from e
            logger.warning(f"Cache get failed: {str(e)}")
            return None
    
    def _update_metadata(self, cache_key: str, cache_type: str) -> None:
        """Update cache metadata for analytics and optimization"""
        now = time.time()
        if cache_key not in self._metadata:
            self._metadata[cache_key] = {
                'first_access': now,
                'last_access': now,
                'access_count': 1,
                'cache_type': cache_type
            }
        else:
            meta = self._metadata[cache_key]
            meta['last_access'] = now
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['cache_type'] = cache_type  # Update in case it moved between caches
    
    def set(self, file_path: Union[str, BinaryIO], strategy: str, result: Dict[str, Any]) -> bool:
        """
        Cache processing result for a file and strategy.
        
        Args:
            file_path: Path to the file or file-like object
            strategy: Chunking strategy used
            result: Processing result to cache
            
        Returns:
            True if successfully cached, False otherwise
            
        Raises:
            CacheError: If CUSTOM_EXCEPTIONS_AVAILABLE is True and there's an error writing to the cache
        """
        file_id = str(file_path) if isinstance(file_path, str) else "file_object"
        
        try:
            # Validate result format
            if not isinstance(result, dict):
                raise ValueError(f"Result must be a dictionary, got {type(result)}")
                
            file_hash = self.get_hash(file_path)
            cache_key = f"{file_hash}_{strategy}"
            
            # Determine appropriate cache based on result size/characteristics
            cache_type = self._select_cache_for_result(result)
            
            # Store in appropriate cache
            if cache_type == 'lru':
                self._lru_cache[cache_key] = result
            else:  # default to TTL
                self._ttl_cache[cache_key] = result
                
            # Update metadata
            self._update_metadata(cache_key, cache_type)
            logger.debug(f"Cached result for {file_id} with strategy {strategy} in {cache_type} cache")
            
            return True
            
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "strategy": strategy,
                "traceback": traceback.format_exc()
            }
            
            if CUSTOM_EXCEPTIONS_AVAILABLE:
                logger.warning(f"Failed to cache result for {file_id}: {str(e)}")
                raise CacheError(
                    f"Failed to store in cache: {str(e)}", 
                    operation="set",
                    file_path=file_id if isinstance(file_path, str) else None,
                    details=error_details
                ) from e
            else:
                # Log the error and return False for backward compatibility
                logger.warning(f"Cache set failed: {str(e)}")
                return False

    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            cache_type: Which cache to clear ('ttl', 'lru', or None for all)
        """
        try:
            if cache_type is None or cache_type == 'ttl':
                self._ttl_cache.clear()
                logger.debug("TTL cache cleared")
                
            if cache_type is None or cache_type == 'lru':
                self._lru_cache.clear()
                logger.debug("LRU cache cleared")
                
            if cache_type is None:
                self._metadata.clear()
                logger.debug("All caches cleared")
        except Exception as e:
            logger.warning(f"Cache clear failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and diagnostics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            now = time.time()
            stats = {
                "ttl_cache_size": len(self._ttl_cache),
                "ttl_cache_maxsize": self._ttl_max_size,
                "lru_cache_size": len(self._lru_cache),
                "lru_cache_maxsize": self._lru_max_size,
                "total_keys": len(self._metadata),
                "access_counts": {}
            }
            
            # Calculate age of cache entries and most accessed keys
            age_sum = 0
            oldest_key = None
            oldest_age = 0
            most_accessed_key = None
            most_accessed_count = 0
            
            for key, meta in self._metadata.items():
                age = now - meta.get('first_access', now)
                age_sum += age
                
                if age > oldest_age:
                    oldest_age = age
                    oldest_key = key
                
                access_count = meta.get('access_count', 0)
                if access_count > most_accessed_count:
                    most_accessed_count = access_count
                    most_accessed_key = key
                    
                # Truncate key for logging
                short_key = key[:15] + "..." if len(key) > 15 else key
                stats["access_counts"][short_key] = access_count
            
            if self._metadata:
                stats["avg_entry_age_seconds"] = age_sum / len(self._metadata)
                stats["oldest_entry_age_seconds"] = oldest_age
                stats["most_accessed_count"] = most_accessed_count
                
            return stats
        except Exception as e:
            logger.warning(f"Failed to generate cache stats: {str(e)}")
            return {"error": str(e)}
    
    def configure(self, ttl_seconds: Optional[int] = None, ttl_max_size: Optional[int] = None, 
                 lru_max_size: Optional[int] = None) -> None:
        """
        Configure cache parameters.
        
        Args:
            ttl_seconds: Time-to-live in seconds for TTL cache
            ttl_max_size: Maximum size for TTL cache
            lru_max_size: Maximum size for LRU cache
        """
        try:
            if ttl_seconds is not None:
                self._ttl_seconds = ttl_seconds
                # Need to recreate the cache with new TTL
                old_cache_items = list(self._ttl_cache.items())
                self._ttl_cache = TTLCache(
                    maxsize=ttl_max_size or self._ttl_max_size, 
                    ttl=ttl_seconds
                )
                # Restore items
                for k, v in old_cache_items:
                    self._ttl_cache[k] = v
                    
            if ttl_max_size is not None and ttl_max_size != self._ttl_max_size:
                self._ttl_max_size = ttl_max_size
                # Need to recreate the cache with new size
                old_cache_items = list(self._ttl_cache.items())
                self._ttl_cache = TTLCache(maxsize=ttl_max_size, ttl=self._ttl_seconds)
                # Restore as many items as will fit
                for k, v in old_cache_items:
                    if len(self._ttl_cache) < ttl_max_size:
                        self._ttl_cache[k] = v
                
            if lru_max_size is not None and lru_max_size != self._lru_max_size:
                self._lru_max_size = lru_max_size
                # Need to recreate the cache with new size
                old_cache_items = list(self._lru_cache.items())
                self._lru_cache = LRUCache(maxsize=lru_max_size)
                # Restore as many items as will fit
                for k, v in old_cache_items:
                    if len(self._lru_cache) < lru_max_size:
                        self._lru_cache[k] = v
                        
            logger.debug(f"Cache reconfigured: TTL={self._ttl_seconds}s, "
                         f"TTL size={self._ttl_max_size}, LRU size={self._lru_max_size}")
        except Exception as e:
            logger.error(f"Cache configuration failed: {str(e)}")
            
    def remove(self, file_path: str, strategy: Optional[str] = None) -> bool:
        """
        Remove specific entries from cache.
        
        Args:
            file_path: Path to the file to remove from cache
            strategy: Specific strategy to remove, None for all strategies
            
        Returns:
            True if any entries were removed, False otherwise
        """
        try:
            file_hash = self.get_hash(file_path)
            removed = False
            
            # Find keys to remove (may be multiple if strategy is None)
            keys_to_remove = []
            for cache_dict in [self._ttl_cache, self._lru_cache]:
                for key in list(cache_dict.keys()):
                    if key.startswith(f"{file_hash}_"):
                        if strategy is None or key == f"{file_hash}_{strategy}":
                            keys_to_remove.append(key)
            
            # Remove from both caches and metadata
            for key in keys_to_remove:
                if key in self._ttl_cache:
                    del self._ttl_cache[key]
                    removed = True
                    
                if key in self._lru_cache:
                    del self._lru_cache[key]
                    removed = True
                    
                if key in self._metadata:
                    del self._metadata[key]
                    
            if removed:
                logger.debug(f"Removed {file_path} entries from cache")
                
            return removed
        except Exception as e:
            logger.warning(f"Cache removal failed: {str(e)}")
            return False

# Create a global instance
document_cache = DocumentCache()

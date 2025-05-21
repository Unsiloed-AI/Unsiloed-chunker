"""
Test the document cache implementation.

This script tests the document cache implementation with
the new TTLCache and LRUCache capabilities.
"""
import os
import sys
import time
import json
import logging
from typing import Dict, Any
import unittest
import tempfile

# Add the parent directory to sys.path to import the Unsiloed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Unsiloed.utils.document_cache import DocumentCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDocumentCache(unittest.TestCase):
    """Test the document cache implementation."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        with open(self.temp_file.name, 'w') as f:
            f.write("This is a test document.\nIt has multiple lines.\nLine 3.\nLine 4.\n")
        
        # Create a second temporary file with different content
        self.temp_file2 = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        with open(self.temp_file2.name, 'w') as f:
            f.write("This is another test document.\nWith different content.\n")
        
        # Reset the document cache singleton for testing
        self.document_cache = DocumentCache()
        self.document_cache._ttl_cache.clear()
        self.document_cache._lru_cache.clear()
        self.document_cache._metadata.clear()
        
        # Configure cache for testing
        self.document_cache.configure(ttl_seconds=1, ttl_max_size=10, lru_max_size=5)
        
    def tearDown(self):
        """Clean up after the test."""
        os.unlink(self.temp_file.name)
        os.unlink(self.temp_file2.name)

    def test_get_hash(self):
        """Test getting a hash of a file."""
        file_hash = self.document_cache.get_hash(self.temp_file.name)
        self.assertIsInstance(file_hash, str)
        self.assertEqual(len(file_hash), 32)  # MD5 hash is 32 characters
        
        # Test that the hash is consistent
        file_hash2 = self.document_cache.get_hash(self.temp_file.name)
        self.assertEqual(file_hash, file_hash2)
        
        # Test that different files have different hashes
        file_hash3 = self.document_cache.get_hash(self.temp_file2.name)
        self.assertNotEqual(file_hash, file_hash3)

    def test_cache_operations(self):
        """Test basic cache operations."""
        # Create test data
        test_data = {
            "chunks": [{"text": "Test chunk 1", "metadata": {}}, 
                      {"text": "Test chunk 2", "metadata": {}}],
            "metadata": {"test": "data"}
        }
        
        # Set cache
        self.document_cache.set(self.temp_file.name, "test_strategy", test_data)
        
        # Get from cache
        cached_data = self.document_cache.get(self.temp_file.name, "test_strategy")
        self.assertIsNotNone(cached_data)
        self.assertEqual(cached_data, test_data)
        
        # Test with different strategy
        different_strategy = self.document_cache.get(self.temp_file.name, "different_strategy")
        self.assertIsNone(different_strategy)
        
        # Test with different file
        different_file = self.document_cache.get(self.temp_file2.name, "test_strategy")
        self.assertIsNone(different_file)
        
    def test_cache_ttl(self):
        """Test that TTL cache expires entries."""
        # Create test data
        small_data = {"text": "Small test data", "chunks": [{"text": "a small chunk"}]}
        
        # Set cache with small data (should go to TTL cache)
        self.document_cache.set(self.temp_file.name, "ttl_test", small_data)
        
        # Verify it's in the cache
        self.assertIsNotNone(self.document_cache.get(self.temp_file.name, "ttl_test"))
        
        # Wait for TTL to expire (we set it to 1 second in setUp)
        time.sleep(1.1)
        
        # Should be expired now
        self.assertIsNone(self.document_cache.get(self.temp_file.name, "ttl_test"))

    def test_cache_lru(self):
        """Test that LRU cache evicts least recently used entries."""
        # Create test data for LRU cache (large data)
        large_data = {
            "chunks": [{"text": "x" * 10000} for _ in range(10)],  # Large chunks
            "metadata": {"test": "data"}
        }
        
        # Set cache for multiple files to exceed LRU capacity
        for i in range(10):  # More than LRU max_size
            # Create a temporary file with unique content
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'_{i}.txt')
            with open(temp_file.name, 'w') as f:
                f.write(f"File {i} content")
            
            # Add to LRU cache
            self.document_cache.set(temp_file.name, "lru_test", large_data)
            os.unlink(temp_file.name)
        
        # Check cache stats
        stats = self.document_cache.get_stats()
        self.assertLessEqual(stats["lru_cache_size"], self.document_cache._lru_max_size)
    
    def test_cache_removal(self):
        """Test removing entries from the cache."""
        # Add data to cache
        test_data = {"text": "Test data", "chunks": [{"text": "a chunk"}]}
        self.document_cache.set(self.temp_file.name, "strategy1", test_data)
        self.document_cache.set(self.temp_file.name, "strategy2", test_data)
        
        # Verify both are in cache
        self.assertIsNotNone(self.document_cache.get(self.temp_file.name, "strategy1"))
        self.assertIsNotNone(self.document_cache.get(self.temp_file.name, "strategy2"))
        
        # Remove specific strategy
        self.document_cache.remove(self.temp_file.name, "strategy1")
        self.assertIsNone(self.document_cache.get(self.temp_file.name, "strategy1"))
        self.assertIsNotNone(self.document_cache.get(self.temp_file.name, "strategy2"))
        
        # Remove all strategies
        self.document_cache.remove(self.temp_file.name)
        self.assertIsNone(self.document_cache.get(self.temp_file.name, "strategy1"))
        self.assertIsNone(self.document_cache.get(self.temp_file.name, "strategy2"))
    
    def test_cache_stats(self):
        """Test getting cache statistics."""
        # Add data to cache
        test_data = {"text": "Test data", "chunks": [{"text": "a chunk"}]}
        self.document_cache.set(self.temp_file.name, "stats_test", test_data)
        
        # Get cache multiple times to increase access count
        for _ in range(3):
            self.document_cache.get(self.temp_file.name, "stats_test")
        
        # Get stats
        stats = self.document_cache.get_stats()
        self.assertGreaterEqual(stats["ttl_cache_size"], 1)
        
        # Check if we have metadata tracking
        self.assertGreaterEqual(len(stats["access_counts"]), 1)
        
        # Should have tracked access counts
        access_counts = stats["access_counts"]
        self.assertTrue(any(count >= 3 for count in access_counts.values()))

if __name__ == "__main__":
    unittest.main()

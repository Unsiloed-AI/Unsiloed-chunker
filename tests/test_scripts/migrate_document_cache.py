"""
Migrate existing document cache to the new TTL/LRU implementation.

This script migrates document processing data from the old
single-dict cache to the new TTL/LRU dual-cache implementation.
"""
import os
import sys
import json
import logging
import time
import argparse
import pickle
from typing import Dict, Any, Tuple

# Add the parent directory to sys.path to import the Unsiloed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Unsiloed.utils.document_cache import document_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_cache_type(data: Dict[str, Any]) -> str:
    """
    Analyze a cache entry to determine which cache it should go to.
    
    Args:
        data: Cache data to analyze
        
    Returns:
        Cache type: 'ttl' or 'lru'
    """
    # Check for large chunks
    if 'chunks' in data and isinstance(data['chunks'], list):
        chunks = data['chunks']
        
        # If many chunks or large chunks, use LRU
        if len(chunks) > 50:
            return 'lru'
        
        # Check total size of chunks
        total_chunk_size = sum(len(chunk.get('text', '')) for chunk in chunks if isinstance(chunk, dict))
        if total_chunk_size > 100000:  # > 100KB
            return 'lru'
            
    # For smaller results or non-chunk data, use TTL
    return 'ttl'


def migrate_cache(old_cache_path: str) -> Tuple[int, int, int]:
    """
    Migrate data from old cache file to the new cache implementation.
    
    Args:
        old_cache_path: Path to the old cache file
        
    Returns:
        Tuple of (total entries, ttl entries, lru entries)
    """
    try:
        # Load the old cache
        if old_cache_path.endswith('.json'):
            with open(old_cache_path, 'r') as f:
                old_cache = json.load(f)
        elif old_cache_path.endswith('.pickle'):
            with open(old_cache_path, 'rb') as f:
                old_cache = pickle.load(f)
        else:
            logger.error(f"Unsupported cache file format: {old_cache_path}")
            return 0, 0, 0
            
        if not isinstance(old_cache, dict):
            logger.error(f"Invalid cache format, expected dict but got {type(old_cache)}")
            return 0, 0, 0
            
        # Statistics
        total_entries = len(old_cache)
        ttl_entries = 0
        lru_entries = 0
        
        # Process each cache entry
        for key, value in old_cache.items():
            # Skip invalid entries
            if not isinstance(value, dict):
                logger.warning(f"Skipping invalid cache entry for key {key}: {type(value)}")
                continue
                
            # Determine which cache to use based on the data
            cache_type = detect_cache_type(value)
            
            if cache_type == 'ttl':
                document_cache._ttl_cache[key] = value
                ttl_entries += 1
            else:  # lru
                document_cache._lru_cache[key] = value
                lru_entries += 1
                
            # Add metadata
            document_cache._metadata[key] = {
                'first_access': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'cache_type': cache_type,
                'migrated': True
            }
            
        logger.info(f"Migration complete: {total_entries} entries total, "
                   f"{ttl_entries} TTL entries, {lru_entries} LRU entries")
        return total_entries, ttl_entries, lru_entries
        
    except Exception as e:
        logger.error(f"Error migrating cache: {str(e)}")
        return 0, 0, 0


def main():
    parser = argparse.ArgumentParser(description='Migrate document cache data')
    parser.add_argument('--old-cache', required=True, 
                       help='Path to the old cache file (.json or .pickle)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify migration by loading a sample document')
    args = parser.parse_args()
    
    if not os.path.exists(args.old_cache):
        logger.error(f"Cache file not found: {args.old_cache}")
        return 1
        
    # Configure the document cache
    document_cache.configure(ttl_seconds=3600, ttl_max_size=500, lru_max_size=200)
    
    # Migrate the cache
    total, ttl, lru = migrate_cache(args.old_cache)
    
    if total == 0:
        logger.error("Migration failed or no entries to migrate")
        return 1
        
    # Print summary
    print(f"\nCache Migration Summary:")
    print(f"  Total entries migrated: {total}")
    print(f"  TTL cache entries: {ttl}")
    print(f"  LRU cache entries: {lru}")
    
    # Optional verification
    if args.verify:
        print("\nVerifying cache by checking stats...")
        stats = document_cache.get_stats()
        print(f"  TTL cache size: {stats['ttl_cache_size']}")
        print(f"  LRU cache size: {stats['lru_cache_size']}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

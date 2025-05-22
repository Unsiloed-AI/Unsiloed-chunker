"""
Memory profiling utility for document processing.

This module provides tools to monitor and analyze memory usage
when processing large documents, helping to identify memory leaks
or inefficient memory usage patterns.
"""
import os
import time
import psutil
import tracemalloc
import logging
from typing import Dict, Any, List, Callable, Optional
import functools
import gc

# Configure logging
logger = logging.getLogger(__name__)

class MemoryProfiler:
    """
    Memory profiler for document processing.
    
    This class:
    1. Tracks memory usage during document processing operations
    2. Provides memory snapshots for detailed analysis
    3. Supports monitoring of peak memory usage
    4. Can be used as a decorator or context manager
    """
    
    def __init__(self, label: str = "memory_profile", enable_tracemalloc: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            label: Label for this profiling session
            enable_tracemalloc: Whether to use tracemalloc for detailed snapshots
        """
        self.label = label
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process(os.getpid())
        self.snapshots = {}
        self.peak_memory = 0
        self.start_memory = 0
        self.end_memory = 0
        self.start_time = 0
        self.end_time = 0
        
    def __enter__(self):
        """Start profiling when entering context."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """End profiling when exiting context."""
        self.end()
        self.report()
        
    def __call__(self, func):
        """Use as a decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
        
    def start(self):
        """Start memory profiling."""
        # Force garbage collection before starting
        gc.collect()
        
        # Record start time
        self.start_time = time.time()
        
        # Record initial memory usage
        mem_info = self.process.memory_info()
        self.start_memory = mem_info.rss
        self.peak_memory = self.start_memory
        
        logger.info(f"[{self.label}] Starting memory profile: {self.start_memory / (1024 * 1024):.2f} MB")
        
        # Start tracemalloc if enabled
        if self.enable_tracemalloc:
            tracemalloc.start()
            self.snapshots["start"] = tracemalloc.take_snapshot()
            
    def snapshot(self, name: str):
        """
        Take a memory snapshot.
        
        Args:
            name: Name of this snapshot
        """
        if not name:
            name = f"snapshot_{len(self.snapshots)}"
            
        # Record memory usage
        mem_info = self.process.memory_info()
        current_memory = mem_info.rss
        
        # Update peak memory if needed
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
        # Take tracemalloc snapshot if enabled
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            self.snapshots[name] = tracemalloc.take_snapshot()
            
        logger.info(f"[{self.label}] Memory snapshot '{name}': {current_memory / (1024 * 1024):.2f} MB")
        return current_memory
        
    def end(self):
        """End memory profiling."""
        # Record end time
        self.end_time = time.time()
        
        # Record final memory usage
        mem_info = self.process.memory_info()
        self.end_memory = mem_info.rss
        
        # Update peak memory if needed
        if self.end_memory > self.peak_memory:
            self.peak_memory = self.end_memory
            
        # Take final tracemalloc snapshot if enabled
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            self.snapshots["end"] = tracemalloc.take_snapshot()
            tracemalloc.stop()
            
        logger.info(f"[{self.label}] Ending memory profile: {self.end_memory / (1024 * 1024):.2f} MB")
        
        # Force garbage collection after ending
        gc.collect()
        
    def report(self):
        """Generate a memory usage report."""
        duration = self.end_time - self.start_time
        memory_change = self.end_memory - self.start_memory
        
        logger.info(f"\n=== Memory Profile Report: {self.label} ===")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Initial memory: {self.start_memory / (1024 * 1024):.2f} MB")
        logger.info(f"Final memory: {self.end_memory / (1024 * 1024):.2f} MB")
        logger.info(f"Peak memory: {self.peak_memory / (1024 * 1024):.2f} MB")
        logger.info(f"Memory change: {memory_change / (1024 * 1024):.2f} MB")
        
        # Report potential memory leak if memory increased significantly
        if memory_change > 10 * 1024 * 1024:  # 10MB threshold
            logger.warning(f"Potential memory leak detected: {memory_change / (1024 * 1024):.2f} MB not released")
            
        # Compare tracemalloc snapshots if available
        if "start" in self.snapshots and "end" in self.snapshots:
            top_stats = self.snapshots["end"].compare_to(self.snapshots["start"], 'lineno')
            logger.info("\nTop 10 memory differences:")
            for stat in top_stats[:10]:
                logger.info(f"{stat}")
                
        return {
            "label": self.label,
            "duration_seconds": duration,
            "initial_memory_mb": self.start_memory / (1024 * 1024),
            "final_memory_mb": self.end_memory / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_change_mb": memory_change / (1024 * 1024)
        }
        
    def compare_snapshots(self, snapshot1: str, snapshot2: str):
        """
        Compare two memory snapshots.
        
        Args:
            snapshot1: Name of first snapshot
            snapshot2: Name of second snapshot
        """
        if not self.enable_tracemalloc or snapshot1 not in self.snapshots or snapshot2 not in self.snapshots:
            logger.error(f"Cannot compare snapshots: missing {snapshot1} or {snapshot2}")
            return
            
        top_stats = self.snapshots[snapshot2].compare_to(self.snapshots[snapshot1], 'lineno')
        
        logger.info(f"\n=== Memory Comparison: {snapshot1} to {snapshot2} ===")
        for stat in top_stats[:10]:
            logger.info(f"{stat}")
            
    def get_current_memory_mb(self):
        """Get current memory usage in MB."""
        mem_info = self.process.memory_info()
        return mem_info.rss / (1024 * 1024)


# Decorator for memory profiling a function
def profile_memory(label: Optional[str] = None):
    """
    Decorator for memory profiling a function.
    
    Args:
        label: Custom label for this profiling session
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_label = label or f"{func.__module__}.{func.__name__}"
            with MemoryProfiler(label=func_label) as profiler:
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


# Function to profile a specific code block
def profile_block(label: str, func: Callable, *args, **kwargs):
    """
    Profile a specific code block or function call.
    
    Args:
        label: Label for this profiling session
        func: Function to profile
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of the function call
    """
    with MemoryProfiler(label=label) as profiler:
        result = func(*args, **kwargs)
        return result

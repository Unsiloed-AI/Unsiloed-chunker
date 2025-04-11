"""
zChunk - A text chunking library for RAG applications using LLM logprobs.
"""

from .zchunker import LlamaChunker, ChunkerConfig, ChunkResult
from .constants import DEFAULT_BIG_SPLIT_TOKEN, DEFAULT_SMALL_SPLIT_TOKEN

__all__ = [
    "LlamaChunker",
    "ChunkerConfig",
    "ChunkResult",
    "DEFAULT_BIG_SPLIT_TOKEN",
    "DEFAULT_SMALL_SPLIT_TOKEN",
]

__version__ = "0.1.0"

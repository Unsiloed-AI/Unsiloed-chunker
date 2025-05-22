"""
Custom exceptions for the Unsiloed-chunker library.

This module provides specialized exception classes to help with:
1. Better error categorization
2. More informative error messages
3. Improved error handling and recovery
4. Clearer debugging information
"""
from typing import Optional, Dict, Any, Union


class UnsiloedError(Exception):
    """Base exception class for all Unsiloed-chunker errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class DocumentProcessingError(UnsiloedError):
    """Error during document processing operations."""
    pass


class ExtractionError(DocumentProcessingError):
    """Error during text extraction from documents."""
    
    def __init__(
        self, 
        message: str, 
        file_type: str,
        extraction_method: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.file_type = file_type
        self.extraction_method = extraction_method
        self.file_path = file_path
        super().__init__(
            message, 
            details={
                "file_type": file_type,
                "extraction_method": extraction_method,
                "file_path": file_path,
                **(details or {})
            }
        )


class PDFExtractionError(ExtractionError):
    """Error during PDF text extraction."""
    
    def __init__(
        self, 
        message: str, 
        extraction_method: Optional[str] = None,
        file_path: Optional[str] = None,
        page_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.page_number = page_number
        super().__init__(
            message, 
            file_type="pdf", 
            extraction_method=extraction_method,
            file_path=file_path,
            details={
                "page_number": page_number,
                **(details or {})
            }
        )


class DocxExtractionError(ExtractionError):
    """Error during DOCX text extraction."""
    
    def __init__(
        self, 
        message: str, 
        extraction_method: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message, 
            file_type="docx", 
            extraction_method=extraction_method,
            file_path=file_path,
            details=details
        )


class PptxExtractionError(ExtractionError):
    """Error during PPTX text extraction."""
    
    def __init__(
        self, 
        message: str, 
        extraction_method: Optional[str] = None,
        file_path: Optional[str] = None,
        slide_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.slide_number = slide_number
        super().__init__(
            message, 
            file_type="pptx", 
            extraction_method=extraction_method,
            file_path=file_path,
            details={
                "slide_number": slide_number,
                **(details or {})
            }
        )


class ChunkingError(DocumentProcessingError):
    """Error during text chunking operations."""
    
    def __init__(
        self, 
        message: str, 
        strategy: str,
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.strategy = strategy
        self.text_length = text_length
        super().__init__(
            message, 
            details={
                "strategy": strategy,
                "text_length": text_length,
                **(details or {})
            }
        )


class CacheError(UnsiloedError):
    """Error related to document cache operations."""
    
    def __init__(
        self, 
        message: str, 
        operation: str,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.file_path = file_path
        super().__init__(
            message, 
            details={
                "operation": operation,
                "file_path": file_path,
                **(details or {})
            }
        )


class MemoryError(UnsiloedError):
    """Error related to memory limits during processing."""
    
    def __init__(
        self, 
        message: str, 
        current_usage_mb: Optional[float] = None,
        limit_mb: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.current_usage_mb = current_usage_mb
        self.limit_mb = limit_mb
        super().__init__(
            message, 
            details={
                "current_usage_mb": current_usage_mb,
                "limit_mb": limit_mb,
                **(details or {})
            }
        )


class FileFormatError(DocumentProcessingError):
    """Error related to unsupported or invalid file formats."""
    
    def __init__(
        self, 
        message: str, 
        file_type: str,
        file_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.file_type = file_type
        self.file_path = file_path
        super().__init__(
            message, 
            details={
                "file_type": file_type,
                "file_path": file_path,
                **(details or {})
            }
        )


class UnsupportedOperationError(UnsiloedError):
    """Error for operations that are not supported in the current configuration."""
    pass


class DependencyError(UnsiloedError):
    """Error related to missing or incompatible dependencies."""
    
    def __init__(
        self, 
        message: str, 
        dependency: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.dependency = dependency
        super().__init__(
            message, 
            details={
                "dependency": dependency,
                **(details or {})
            }
        )

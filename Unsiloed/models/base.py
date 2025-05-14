"""
Base classes for model providers.
This module defines the abstract interfaces that all model providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """
    Abstract base class for model providers.
    All model providers must implement these methods.
    """

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the model provider.

        Args:
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific configuration
        """
        pass

    @abstractmethod
    def get_client(self):
        """
        Get the client for the model provider.

        Returns:
            The client object for the provider
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using the model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for models that support it
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_structured_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_schema: Dict[str, Any] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) using the model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt for models that support it
            output_schema: Schema for the expected output
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated structured output as a dictionary
        """
        pass

    @abstractmethod
    def process_image(
        self, image_path: str, prompt: str, **kwargs
    ) -> str:
        """
        Process an image using the model.

        Args:
            image_path: Path to the image file
            prompt: Prompt to send with the image
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text from the image
        """
        pass


class ModelProviderFactory:
    """
    Factory class for creating model providers.
    """

    _providers = {}

    @classmethod
    def register_provider(cls, name: str, provider_class):
        """
        Register a model provider.

        Args:
            name: Name of the provider
            provider_class: Provider class
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered model provider: {name}")

    @classmethod
    def create_provider(cls, name: str, **kwargs) -> ModelProvider:
        """
        Create a model provider instance.

        Args:
            name: Name of the provider
            **kwargs: Provider-specific configuration

        Returns:
            ModelProvider instance
        """
        if name not in cls._providers:
            raise ValueError(f"Unknown model provider: {name}")

        logger.info(f"Creating model provider: {name}")
        return cls._providers[name](**kwargs)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get a list of available providers.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

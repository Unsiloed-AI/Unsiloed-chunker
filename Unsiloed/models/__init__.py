"""
Model providers package.
"""

from Unsiloed.models.base import ModelProvider, ModelProviderFactory
from Unsiloed.models.openai_provider import OpenAIProvider

# Import optional providers
try:
    from Unsiloed.models.anthropic_provider import AnthropicProvider
except ImportError:
    pass

try:
    from Unsiloed.models.huggingface_provider import HuggingFaceProvider
except ImportError:
    pass

try:
    from Unsiloed.models.local_provider import LocalLLMProvider
except ImportError:
    pass

# Export the factory and base classes
__all__ = ["ModelProvider", "ModelProviderFactory", "get_provider"]


def get_provider(provider_name: str = None, **kwargs):
    """
    Get a model provider instance.
    
    Args:
        provider_name: Name of the provider (default: from environment or "openai")
        **kwargs: Provider-specific configuration
        
    Returns:
        ModelProvider instance
    """
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # If provider_name is not specified, try to get it from environment
    if provider_name is None:
        provider_name = os.environ.get("UNSILOED_MODEL_PROVIDER", "openai")
        
    # Get available providers
    available_providers = ModelProviderFactory.get_available_providers()
    
    if provider_name not in available_providers:
        logger.warning(f"Provider '{provider_name}' not available. Available providers: {available_providers}")
        logger.info(f"Falling back to OpenAI provider")
        provider_name = "openai"
        
    # Create the provider
    return ModelProviderFactory.create_provider(provider_name, **kwargs)

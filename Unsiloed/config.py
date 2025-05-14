"""
Configuration module for Unsiloed.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration class for Unsiloed.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "model_provider": "openai",
        "openai": {
            "api_key": None,
            "model": "gpt-4o",
            "timeout": 60.0,
            "max_retries": 3,
        },
        "anthropic": {
            "api_key": None,
            "model": "claude-3-opus-20240229",
            "timeout": 60.0,
            "max_retries": 3,
        },
        "huggingface": {
            "api_key": None,
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "api_url": None,
            "timeout": 60.0,
        },
        "local": {
            "model_path": None,
            "n_ctx": 2048,
            "n_gpu_layers": -1,
            "verbose": False,
        },
    }

    def __init__(self):
        """
        Initialize the configuration.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_from_env()

    def _load_from_env(self):
        """
        Load configuration from environment variables.
        """
        # Model provider
        self.config["model_provider"] = os.environ.get(
            "UNSILOED_MODEL_PROVIDER", self.config["model_provider"]
        )

        # OpenAI configuration
        self.config["openai"]["api_key"] = os.environ.get(
            "OPENAI_API_KEY", self.config["openai"]["api_key"]
        )
        self.config["openai"]["model"] = os.environ.get(
            "OPENAI_MODEL", self.config["openai"]["model"]
        )
        self.config["openai"]["timeout"] = float(
            os.environ.get("OPENAI_TIMEOUT", self.config["openai"]["timeout"])
        )
        self.config["openai"]["max_retries"] = int(
            os.environ.get("OPENAI_MAX_RETRIES", self.config["openai"]["max_retries"])
        )

        # Anthropic configuration
        self.config["anthropic"]["api_key"] = os.environ.get(
            "ANTHROPIC_API_KEY", self.config["anthropic"]["api_key"]
        )
        self.config["anthropic"]["model"] = os.environ.get(
            "ANTHROPIC_MODEL", self.config["anthropic"]["model"]
        )
        self.config["anthropic"]["timeout"] = float(
            os.environ.get("ANTHROPIC_TIMEOUT", self.config["anthropic"]["timeout"])
        )
        self.config["anthropic"]["max_retries"] = int(
            os.environ.get(
                "ANTHROPIC_MAX_RETRIES", self.config["anthropic"]["max_retries"]
            )
        )

        # Hugging Face configuration
        self.config["huggingface"]["api_key"] = os.environ.get(
            "HUGGINGFACE_API_KEY", self.config["huggingface"]["api_key"]
        )
        self.config["huggingface"]["model"] = os.environ.get(
            "HUGGINGFACE_MODEL", self.config["huggingface"]["model"]
        )
        self.config["huggingface"]["api_url"] = os.environ.get(
            "HUGGINGFACE_API_URL", self.config["huggingface"]["api_url"]
        )
        self.config["huggingface"]["timeout"] = float(
            os.environ.get("HUGGINGFACE_TIMEOUT", self.config["huggingface"]["timeout"])
        )

        # Local LLM configuration
        self.config["local"]["model_path"] = os.environ.get(
            "LOCAL_MODEL_PATH", self.config["local"]["model_path"]
        )
        self.config["local"]["n_ctx"] = int(
            os.environ.get("LOCAL_N_CTX", self.config["local"]["n_ctx"])
        )
        self.config["local"]["n_gpu_layers"] = int(
            os.environ.get("LOCAL_N_GPU_LAYERS", self.config["local"]["n_gpu_layers"])
        )
        self.config["local"]["verbose"] = (
            os.environ.get("LOCAL_VERBOSE", "").lower() == "true"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

    def get_provider_config(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration for a specific provider.

        Args:
            provider_name: Provider name (if None, use the default provider)

        Returns:
            Provider configuration
        """
        if provider_name is None:
            provider_name = self.config["model_provider"]

        if provider_name not in self.config:
            logger.warning(f"Provider '{provider_name}' not found in configuration")
            return {}

        return self.config[provider_name]


# Create a singleton instance
config = Config()

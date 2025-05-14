"""
Local LLM provider implementation using llama.cpp.
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
import logging
import subprocess
import tempfile
import requests
from pathlib import Path

from Unsiloed.models.base import ModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)

# Check if llama-cpp-python package is available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python package not available. Install with 'pip install llama-cpp-python'")


class LocalLLMProvider(ModelProvider):
    """
    Local LLM provider implementation using llama.cpp.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 means use all available GPU layers
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the Local LLM provider.

        Args:
            model_path: Path to the model file
            n_ctx: Context size
            n_gpu_layers: Number of GPU layers to use (-1 = all)
            verbose: Whether to enable verbose output
            **kwargs: Additional provider-specific configuration
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python package not available. Install with 'pip install llama-cpp-python'"
            )
            
        self.model_path = model_path
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.kwargs = kwargs
        self.client = None

    def get_client(self):
        """
        Get the llama.cpp client.

        Returns:
            Llama instance
        """
        if self.client is None:
            logger.debug(f"Loading model from {self.model_path}")
            try:
                self.client = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                    **self.kwargs
                )
                logger.debug("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                raise
                
        return self.client

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using the local LLM.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        client = self.get_client()
        
        # Combine system prompt and user prompt if both are provided
        if system_prompt:
            # Format depends on the model type, this is a common format
            full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
        else:
            # Basic prompt format for Llama models
            full_prompt = f"<s>[INST] {prompt} [/INST]"
        
        try:
            output = client.create_completion(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Extract the generated text
            if "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["text"]
            else:
                return ""
        except Exception as e:
            logger.error(f"Error generating text with local LLM: {str(e)}")
            raise

    def generate_structured_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_schema: Dict[str, Any] = None,
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) using the local LLM.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            output_schema: Schema for the expected output
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated structured output as a dictionary
        """
        # Local LLMs don't have native JSON mode,
        # so we need to instruct the model to return JSON
        
        if output_schema:
            schema_str = json.dumps(output_schema, indent=2)
            json_instruction = f"Return your response as a JSON object with the following schema:\n{schema_str}\n\nYour response should be valid JSON and nothing else."
        else:
            json_instruction = "Return your response as a valid JSON object and nothing else."
        
        enhanced_prompt = f"{prompt}\n\n{json_instruction}"
        
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt} You must return your response as valid JSON and nothing else."
        else:
            enhanced_system_prompt = "You must return your response as valid JSON and nothing else."
        
        try:
            response_text = self.generate_text(
                prompt=enhanced_prompt,
                system_prompt=enhanced_system_prompt,
                temperature=temperature,
                **kwargs
            )
            
            # Extract JSON from the response
            # Sometimes the model might include markdown code block markers
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()
            else:
                json_text = response_text.strip()
            
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error generating structured output with local LLM: {str(e)}")
            raise

    def process_image(
        self, image_path: str, prompt: str, **kwargs
    ) -> str:
        """
        Process an image using the local LLM.

        Args:
            image_path: Path to the image file
            prompt: Prompt to send with the image
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text from the image
        """
        # Check if this is a multimodal model
        if not any(mm_model in self.model_path.lower() for mm_model in ["llava", "bakllava", "multimodal"]):
            logger.error(f"Model {self.model_path} is not a multimodal model capable of processing images")
            raise ValueError(f"Model {self.model_path} is not a multimodal model capable of processing images")
        
        try:
            # For multimodal models in llama.cpp, we need to use the image_path parameter
            client = self.get_client()
            
            output = client.create_completion(
                prompt=prompt,
                image_path=image_path,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.0),
                **kwargs
            )
            
            # Extract the generated text
            if "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["text"]
            else:
                return ""
        except Exception as e:
            logger.error(f"Error processing image with local LLM: {str(e)}")
            raise


# Register the Local LLM provider with the factory
ModelProviderFactory.register_provider("local", LocalLLMProvider)

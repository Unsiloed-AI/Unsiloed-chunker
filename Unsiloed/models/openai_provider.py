"""
OpenAI model provider implementation.
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
import logging
from openai import OpenAI
import numpy as np
import cv2

from Unsiloed.models.base import ModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)


class OpenAIProvider(ModelProvider):
    """
    OpenAI model provider implementation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: Model to use (default: gpt-4o)
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retries for API calls
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided and not found in environment")
            raise ValueError("OpenAI API key is required")

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = None

    def get_client(self):
        """
        Get the OpenAI client.

        Returns:
            OpenAI client instance
        """
        if self.client is None:
            logger.debug("Creating new OpenAI client")
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            
            # Test the client by listing available models
            try:
                models = self.client.models.list()
                if models and hasattr(models, "data") and len(models.data) > 0:
                    logger.debug(
                        f"OpenAI client initialized successfully, available models: {len(models.data)}"
                    )
                else:
                    logger.warning("OpenAI client initialized but returned no models")
            except Exception as e:
                logger.error(f"Error testing OpenAI client: {str(e)}")
                
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
        Generate text using the OpenAI model.

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
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
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
        Generate structured output (JSON) using the OpenAI model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            output_schema: Schema for the expected output
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated structured output as a dictionary
        """
        client = self.get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                **kwargs
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating structured output with OpenAI: {str(e)}")
            raise

    def encode_image_to_base64(self, image_path):
        """
        Encode an image to base64.

        Args:
            image_path: Path to the image file or numpy array

        Returns:
            Base64 encoded string of the image
        """
        logger.debug("Encoding image to base64")

        # Handle numpy array (from CV2)
        if isinstance(image_path, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image_path)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer).decode("utf-8")

        # Handle file path
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def process_image(
        self, image_path: str, prompt: str, **kwargs
    ) -> str:
        """
        Process an image using the OpenAI model.

        Args:
            image_path: Path to the image file
            prompt: Prompt to send with the image
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text from the image
        """
        client = self.get_client()
        
        # Encode the image to base64
        base64_image = self.encode_image_to_base64(image_path)
        
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                **kwargs
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error processing image with OpenAI: {str(e)}")
            raise


# Register the OpenAI provider with the factory
ModelProviderFactory.register_provider("openai", OpenAIProvider)

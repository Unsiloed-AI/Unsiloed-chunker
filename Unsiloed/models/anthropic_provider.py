"""
Anthropic model provider implementation.
"""

import os
import json
import base64
from typing import Dict, List, Any, Optional, Union
import logging
import requests
from io import BytesIO

from Unsiloed.models.base import ModelProvider, ModelProviderFactory

logger = logging.getLogger(__name__)

# Check if anthropic package is available
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic package not available. Install with 'pip install anthropic'")


class AnthropicProvider(ModelProvider):
    """
    Anthropic model provider implementation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        timeout: float = 60.0,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (if None, will try to get from environment)
            model: Model to use (default: claude-3-opus-20240229)
            timeout: Timeout for API calls in seconds
            max_retries: Maximum number of retries for API calls
            **kwargs: Additional provider-specific configuration
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not available. Install with 'pip install anthropic'"
            )
            
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("Anthropic API key not provided and not found in environment")
            raise ValueError("Anthropic API key is required")

        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = None

    def get_client(self):
        """
        Get the Anthropic client.

        Returns:
            Anthropic client instance
        """
        if self.client is None:
            logger.debug("Creating new Anthropic client")
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
            
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
        Generate text using the Anthropic model.

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
        
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {str(e)}")
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
        Generate structured output (JSON) using the Anthropic model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            output_schema: Schema for the expected output
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated structured output as a dictionary
        """
        # Anthropic doesn't have a native JSON mode like OpenAI,
        # so we need to instruct it to return JSON in the prompt
        
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
            logger.error(f"Error generating structured output with Anthropic: {str(e)}")
            raise

    def process_image(
        self, image_path: str, prompt: str, **kwargs
    ) -> str:
        """
        Process an image using the Anthropic model.

        Args:
            image_path: Path to the image file
            prompt: Prompt to send with the image
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text from the image
        """
        client = self.get_client()
        
        try:
            # Read the image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Create a message with the image
            message = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 4000),
                temperature=kwargs.get("temperature", 0.0),
                system=kwargs.get("system_prompt", ""),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(image_data).decode("utf-8")}}
                        ]
                    }
                ]
            )
            
            return message.content[0].text
        except Exception as e:
            logger.error(f"Error processing image with Anthropic: {str(e)}")
            raise


# Register the Anthropic provider with the factory
ModelProviderFactory.register_provider("anthropic", AnthropicProvider)

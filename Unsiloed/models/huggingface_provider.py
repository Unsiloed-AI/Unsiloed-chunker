"""
Hugging Face model provider implementation.
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


class HuggingFaceProvider(ModelProvider):
    """
    Hugging Face model provider implementation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        api_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Initialize the Hugging Face provider.

        Args:
            api_key: Hugging Face API key (if None, will try to get from environment)
            model: Model to use (default: mistralai/Mistral-7B-Instruct-v0.2)
            api_url: API URL (if None, will use the Hugging Face Inference API)
            timeout: Timeout for API calls in seconds
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        if not self.api_key:
            logger.error("Hugging Face API key not provided and not found in environment")
            raise ValueError("Hugging Face API key is required")

        self.model = model
        self.timeout = timeout
        
        # If api_url is provided, use it; otherwise, use the Hugging Face Inference API
        self.api_url = api_url or f"https://api-inference.huggingface.co/models/{model}"
        
        # Check if transformers is available for local inference
        try:
            import transformers
            self.transformers_available = True
        except ImportError:
            self.transformers_available = False
            logger.warning("Transformers package not available. Install with 'pip install transformers'")
        
        self.client = None
        self.local_model = None

    def get_client(self):
        """
        Get the Hugging Face client.
        For Hugging Face, this is just the API key and URL.

        Returns:
            Dictionary with API key and URL
        """
        if self.client is None:
            self.client = {
                "api_key": self.api_key,
                "api_url": self.api_url
            }
        return self.client

    def _call_hf_api(self, payload):
        """
        Call the Hugging Face Inference API.

        Args:
            payload: Payload to send to the API

        Returns:
            API response
        """
        client = self.get_client()
        headers = {"Authorization": f"Bearer {client['api_key']}"}
        
        try:
            response = requests.post(
                client["api_url"],
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
            raise

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using the Hugging Face model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        # Combine system prompt and user prompt if both are provided
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        # Prepare the payload
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
        }
        
        try:
            response = self._call_hf_api(payload)
            
            # Handle different response formats
            if isinstance(response, list) and len(response) > 0:
                if "generated_text" in response[0]:
                    return response[0]["generated_text"]
                else:
                    return str(response[0])
            elif isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"]
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error generating text with Hugging Face: {str(e)}")
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
        Generate structured output (JSON) using the Hugging Face model.

        Args:
            prompt: The prompt to send to the model
            system_prompt: System prompt for the model
            output_schema: Schema for the expected output
            temperature: Temperature for generation (0.0 = deterministic)
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated structured output as a dictionary
        """
        # Hugging Face models don't have native JSON mode,
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
            logger.error(f"Error generating structured output with Hugging Face: {str(e)}")
            raise

    def process_image(
        self, image_path: str, prompt: str, **kwargs
    ) -> str:
        """
        Process an image using the Hugging Face model.

        Args:
            image_path: Path to the image file
            prompt: Prompt to send with the image
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text from the image
        """
        # For image processing, we need to use a multimodal model
        # Check if the model is a multimodal model
        if not any(mm_model in self.model.lower() for mm_model in ["llava", "idefics", "blip", "vit"]):
            logger.warning(f"Model {self.model} may not be a multimodal model capable of processing images")
        
        try:
            # Read the image file
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Encode the image to base64
            encoded_image = base64.b64encode(image_data).decode("utf-8")
            
            # Prepare the payload
            payload = {
                "inputs": {
                    "image": encoded_image,
                    "prompt": prompt
                },
                "parameters": {
                    **kwargs
                }
            }
            
            response = self._call_hf_api(payload)
            
            # Handle different response formats
            if isinstance(response, list) and len(response) > 0:
                if "generated_text" in response[0]:
                    return response[0]["generated_text"]
                else:
                    return str(response[0])
            elif isinstance(response, dict) and "generated_text" in response:
                return response["generated_text"]
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error processing image with Hugging Face: {str(e)}")
            raise


# Register the Hugging Face provider with the factory
ModelProviderFactory.register_provider("huggingface", HuggingFaceProvider)

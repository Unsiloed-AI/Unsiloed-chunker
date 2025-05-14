from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import logging
from openai import OpenAI
import anthropic
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    @abstractmethod
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Get completion from the model"""
        pass
    
    @abstractmethod
    def get_structured_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Get structured completion (JSON) from the model"""
        pass

class OpenAIProvider(ModelProvider):
    """OpenAI model provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content
    
    def get_structured_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs
        )
        return response.choices[0].message.content

class AnthropicProvider(ModelProvider):
    """Anthropic model provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
    
    def get_structured_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not system_prompt:
            system_prompt = "You are a helpful assistant that responds in valid JSON format."
        else:
            system_prompt += " Respond in valid JSON format."
            
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

class HuggingFaceProvider(ModelProvider):
    """HuggingFace model provider implementation"""
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    def get_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 1000),
            temperature=kwargs.get("temperature", 0.7),
            **kwargs
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_structured_completion(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not system_prompt:
            system_prompt = "You are a helpful assistant that responds in valid JSON format."
        else:
            system_prompt += " Respond in valid JSON format."
            
        return self.get_completion(prompt, system_prompt, **kwargs)

def get_model_provider(provider_type: str, **kwargs) -> ModelProvider:
    """Factory function to get the appropriate model provider"""
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "huggingface": HuggingFaceProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    return providers[provider_type](**kwargs) 
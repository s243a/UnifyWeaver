"""
Extended Provider Support for Agent-Based RAG
Adds support for OpenAI, Mistral, Groq, Together AI, and other providers
"""

import os
import json
import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class OpenAIProvider:
    """OpenAI API support (GPT-3.5, GPT-4, etc.)"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None, 
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class MistralProvider:
    """Mistral AI API support"""
    
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using Mistral API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise

class GroqProvider:
    """Groq API support for fast inference"""
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using Groq API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

class TogetherProvider:
    """Together AI API support for open source models"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3-70b-chat-hf"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.together.xyz/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using Together API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Together API error: {e}")
            raise

class DeepSeekProvider:
    """DeepSeek API support"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise

class XAIProvider:
    """X.AI (Grok) API support"""
    
    def __init__(self, api_key: str, model: str = "grok-beta"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.x.ai/v1/chat/completions"
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using X.AI API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"X.AI API error: {e}")
            raise

class ReplicateProvider:
    """Replicate API support for various open source models"""
    
    def __init__(self, api_key: str, model: str = "meta/llama-2-70b-chat"):
        self.api_key = api_key
        self.model = model
    
    async def complete(self, prompt: str, system_prompt: str = None,
                      max_tokens: int = 2000, temperature: float = 0.7) -> str:
        """Generate completion using Replicate API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Replicate uses a different API structure
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "version": self.model,
            "input": {
                "prompt": full_prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            # Create prediction
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            prediction = response.json()
            
            # Poll for result
            result_url = prediction["urls"]["get"]
            while True:
                result = requests.get(result_url, headers=headers).json()
                if result["status"] == "succeeded":
                    return "".join(result["output"])
                elif result["status"] == "failed":
                    raise Exception(f"Prediction failed: {result.get('error')}")
                
                import time
                time.sleep(1)
        except Exception as e:
            logger.error(f"Replicate API error: {e}")
            raise

# Provider factory
def get_provider(provider_name: str, api_key: str, model: str):
    """Factory function to get the appropriate provider"""
    providers = {
        "openai": OpenAIProvider,
        "mistral": MistralProvider,
        "groq": GroqProvider,
        "together": TogetherProvider,
        "deepseek": DeepSeekProvider,
        "xai": XAIProvider,
        "grok": XAIProvider,  # Alias
        "replicate": ReplicateProvider
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class:
        return provider_class(api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

# Extension for the orchestrator agents
class UniversalAgent:
    """Universal agent that can use any provider"""
    
    def __init__(self, role_config: Dict):
        self.provider_name = role_config["provider"]
        self.model = role_config["model"]
        self.api_key = role_config["api_key"]
        self.max_tokens = role_config.get("max_tokens", 2000)
        
        # Get the appropriate provider
        if self.provider_name in ["openai", "mistral", "groq", "together", 
                                  "deepseek", "xai", "grok", "replicate"]:
            self.provider = get_provider(self.provider_name, self.api_key, self.model)
        else:
            # Fall back to existing providers
            self.provider = None
    
    async def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate using the configured provider"""
        if self.provider:
            return await self.provider.complete(
                prompt, 
                system_prompt,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Provider {self.provider_name} not configured")
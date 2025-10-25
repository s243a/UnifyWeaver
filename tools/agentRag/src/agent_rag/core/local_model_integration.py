#!/usr/bin/env python3
"""
Enhanced Local Model Integration with Platform Detection
Supports both Ollama and llama-cpp-python based on environment
Place in: src/agent_rag/core/local_model_integration.py
"""

import os
import sys
import json
import platform
import subprocess
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import requests
from time import time

logger = logging.getLogger(__name__)

@dataclass
class LocalModelConfig:
    """Configuration for local model providers"""
    provider: str  # 'ollama' or 'llama.cpp'
    model_name: Optional[str] = None  # For Ollama
    model_path: Optional[str] = None  # For llama.cpp
    server_url: Optional[str] = None
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.7
    max_tokens: int = 512


class PlatformDetector:
    """Detect runtime environment and recommend appropriate provider"""
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """Detect platform and return configuration"""
        system = platform.system().lower()
        is_cygwin = 'cygwin' in system or 'cygwin' in platform.platform().lower()
        is_termux = (
            os.environ.get('PREFIX') == '/data/data/com.termux/files/usr' or
            'termux' in platform.platform().lower()
        )
        is_wsl = 'microsoft' in platform.release().lower()
        
        # Determine best provider
        if is_cygwin or is_termux:
            provider = 'llama.cpp'
            reason = 'Cygwin/Termux detected - Ollama not supported'
        elif is_wsl:
            provider = 'ollama'
            reason = 'WSL detected - Ollama should work'
        else:
            provider = 'ollama'
            reason = f'{platform.system()} detected - Ollama recommended'
        
        return {
            'system': system,
            'is_cygwin': is_cygwin,
            'is_termux': is_termux,
            'is_wsl': is_wsl,
            'recommended_provider': provider,
            'reason': reason
        }


class OllamaProvider:
    """Provider for Ollama-based inference"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.base_url = config.server_url or "http://localhost:11434"
        self.model = config.model_name or "phi3"
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.ok:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
        except:
            pass
        return []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            if response.ok:
                return response.json()['response']
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
        
        return ""
    
    def score_chunk(self, query: str, chunk: str) -> float:
        """Score relevance of a chunk to query"""
        prompt = f"""Score the relevance of this text to the query on a scale of 0-10.
Query: {query}
Text: {chunk[:500]}
Reply with just a number 0-10:"""
        
        response = self.generate(prompt, max_tokens=10, temperature=0.1)
        try:
            score = float(response.strip().split()[0])
            return min(max(score / 10.0, 0.0), 1.0)
        except:
            return 0.0


class LlamaCppProvider:
    """Provider for llama-cpp-python based inference"""
    
    def __init__(self, config: LocalModelConfig):
        self.config = config
        self.llm = None
        self.server_url = config.server_url or "http://localhost:8080"
        
        # Try to import and initialize
        try:
            from llama_cpp import Llama
            if config.model_path and os.path.exists(config.model_path):
                self.llm = Llama(
                    model_path=config.model_path,
                    n_ctx=config.n_ctx,
                    n_threads=config.n_threads,
                    n_gpu_layers=config.n_gpu_layers,
                    verbose=False
                )
                logger.info(f"Loaded model: {config.model_path}")
        except ImportError:
            logger.warning("llama-cpp-python not installed")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def is_available(self) -> bool:
        """Check if llama-cpp is available"""
        if self.llm:
            return True
        
        # Check if server is running
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start_server(self) -> Optional[subprocess.Popen]:
        """Start llama-cpp-python server"""
        if not self.config.model_path:
            return None
        
        try:
            cmd = [
                sys.executable, "-m", "llama_cpp.server",
                "--model", self.config.model_path,
                "--host", "0.0.0.0",
                "--port", "8080",
                "--n_ctx", str(self.config.n_ctx)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Started llama-cpp-python server")
            return process
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama-cpp"""
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Direct inference if model loaded
        if self.llm:
            try:
                response = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=kwargs.get('stop', [])
                )
                return response['choices'][0]['text']
            except Exception as e:
                logger.error(f"Generation failed: {e}")
        
        # Fall back to server API
        try:
            response = requests.post(
                f"{self.server_url}/v1/completions",
                json={
                    'prompt': prompt,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                },
                timeout=30
            )
            if response.ok:
                return response.json()['choices'][0]['text']
        except Exception as e:
            logger.error(f"Server generation failed: {e}")
        
        return ""
    
    def score_chunk(self, query: str, chunk: str) -> float:
        """Score relevance of a chunk to query"""
        prompt = f"""Score relevance (0-10):
Query: {query}
Text: {chunk[:300]}
Score:"""
        
        response = self.generate(prompt, max_tokens=5, temperature=0.1)
        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = float(numbers[0])
                return min(max(score / 10.0, 0.0), 1.0)
        except:
            pass
        return 0.0


class LocalModelManager:
    """Unified manager for local model providers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.platform_info = PlatformDetector.detect()
        
        # Determine provider
        provider_name = config.get('local_model_provider')
        if not provider_name:
            provider_name = self.platform_info['recommended_provider']
            logger.info(f"Auto-selected provider: {provider_name}")
            logger.info(f"Reason: {self.platform_info['reason']}")
        
        # Initialize provider
        model_config = LocalModelConfig(
            provider=provider_name,
            model_name=config.get('local_retrieval_model', 'phi3'),
            model_path=config.get('local_model_path'),
            n_ctx=config.get('n_ctx', 2048),
            n_threads=config.get('n_threads', 4),
            temperature=config.get('temperature', 0.7)
        )
        
        if provider_name == 'ollama':
            self.provider = OllamaProvider(model_config)
        elif provider_name in ['llama.cpp', 'llama-cpp', 'llama_cpp']:
            self.provider = LlamaCppProvider(model_config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Check availability
        if not self.provider.is_available():
            logger.warning(f"{provider_name} provider not available")
            if provider_name == 'ollama':
                logger.info("Try: ollama pull phi3")
            else:
                logger.info("Check model path and llama-cpp-python installation")
    
    def score_chunks(self, query: str, chunks: List[str]) -> List[Tuple[str, float]]:
        """Score and rank chunks by relevance"""
        scored = []
        for chunk in chunks:
            score = self.provider.score_chunk(query, chunk)
            scored.append((chunk, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def retrieve(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        """Retrieve top-k most relevant chunks"""
        scored = self.score_chunks(query, chunks)
        return [chunk for chunk, score in scored[:top_k]]
    
    def synthesize(self, query: str, context: str) -> str:
        """Synthesize a response from context"""
        prompt = f"""Based on the following context, answer the query.
        
Context:
{context}

Query: {query}

Answer:"""
        
        return self.provider.generate(prompt)
    
    def test_retrieval(self, query: str, chunks: List[str]) -> Dict:
        """Test retrieval with timing"""
        start = time()
        
        results = {
            'query': query,
            'num_chunks': len(chunks),
            'provider': self.provider.__class__.__name__,
            'platform': self.platform_info
        }
        
        try:
            scored = self.score_chunks(query, chunks)
            results['success'] = True
            results['top_chunk'] = scored[0] if scored else None
            results['scores'] = [score for _, score in scored]
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
        
        results['duration'] = time() - start
        return results
    
    def get_info(self) -> Dict:
        """Get information about the current setup"""
        info = {
            'platform': self.platform_info,
            'provider': self.provider.__class__.__name__,
            'available': self.provider.is_available(),
            'config': self.config
        }
        
        if hasattr(self.provider, 'list_models'):
            info['models'] = self.provider.list_models()
        
        return info


# CLI for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test local model integration")
    parser.add_argument("--provider", choices=['ollama', 'llama.cpp'], 
                       help="Force specific provider")
    parser.add_argument("--model", help="Model name (Ollama) or path (llama.cpp)")
    parser.add_argument("--query", default="How does authentication work?",
                       help="Test query")
    parser.add_argument("--info", action="store_true", 
                       help="Show platform and provider info")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Build config
    config = {}
    if args.provider:
        config['local_model_provider'] = args.provider
    if args.model:
        if args.provider == 'llama.cpp':
            config['local_model_path'] = args.model
        else:
            config['local_retrieval_model'] = args.model
    
    # Initialize manager
    manager = LocalModelManager(config)
    
    if args.info:
        info = manager.get_info()
        print(json.dumps(info, indent=2))
    else:
        # Test retrieval
        test_chunks = [
            "Authentication is the process of verifying user identity.",
            "Cats are furry animals that meow.",
            "JWT tokens contain encoded user claims and signatures.",
            "The weather today is sunny and warm."
        ]
        
        print(f"Testing retrieval for: {args.query}")
        print("-" * 50)
        
        results = manager.test_retrieval(args.query, test_chunks)
        
        if results['success']:
            print(f"✅ Success in {results['duration']:.2f}s")
            print(f"Top result: {results['top_chunk'][0][:100]}...")
            print(f"Score: {results['top_chunk'][1]:.2f}")
        else:
            print(f"❌ Failed: {results.get('error')}")
        
        print("\nPlatform info:")
        print(json.dumps(results['platform'], indent=2))
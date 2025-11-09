"""
Configuration management for Agent-Based RAG System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

def get_config():
    """Get the system configuration from environment variables"""
    return {
        "db_path": os.getenv("DB_PATH", "rag_index.db"),
        "vault_path": Path(os.getenv("OBSIDIAN_VAULT", "./vault")),
        "reasoning_mode": os.getenv("REASONING_MODE", "api"),
        "synthesis_mode": os.getenv("SYNTHESIS_MODE", "api"),
        
        "roles": {
            "retriever": {
                "provider": os.getenv("RETRIEVER_PROVIDER", "gemini"),
                "model": os.getenv("RETRIEVER_MODEL", "gemini-2.0-flash-exp"),
                "api_key": os.getenv("RETRIEVER_API_KEY") or os.getenv("GEMINI_API_KEY"),
                "endpoint": os.getenv("RETRIEVER_URL", "http://localhost:8000/gemini-flash-retrieve")
            },
            "global": {
                "provider": os.getenv("GLOBAL_PROVIDER", "perplexity"),
                "model": os.getenv("GLOBAL_MODEL", "llama-3.1-sonar-large-128k-online"),
                "api_key": os.getenv("GLOBAL_API_KEY") or os.getenv("PERPLEXITY_API_KEY"),
                "endpoint": os.getenv("GLOBAL_URL", "https://api.perplexity.ai/chat/completions")
            },
            "synthesizer": {
                "provider": os.getenv("SYNTH_PROVIDER", "perplexity"),
                "model": os.getenv("SYNTH_MODEL", "gpt-5"),
                "api_key": os.getenv("SYNTH_API_KEY") or os.getenv("PERPLEXITY_API_KEY"),
                "endpoint": os.getenv("SYNTH_URL", "https://api.perplexity.ai/chat/completions"),
                "token_limit": int(os.getenv("SYNTH_TOKEN_LIMIT", "4000"))
            },
            "reasoner": {
                "provider": os.getenv("REASON_PROVIDER", "anthropic"),
                "model": os.getenv("REASON_MODEL", "claude-opus-4-1-20250805"),
                "api_key": os.getenv("REASON_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
                "endpoint": os.getenv("REASON_URL", "https://api.anthropic.com/v1/messages"),
                "max_tokens": int(os.getenv("REASON_MAX_TOKENS", "2000"))
            }
        }
    }
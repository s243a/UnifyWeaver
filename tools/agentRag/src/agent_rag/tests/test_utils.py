#!/usr/bin/env python3
"""
Test utilities for Agent-Based RAG System
Quick tests to verify services and configuration
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import json
import requests
import time
from typing import Dict, List, Optional

# Get configuration
try:
    from agent_rag.config import get_config
    config = get_config()
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    config = None

# ANSI color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message: str, status: str = "info"):
    """Print colored status message"""
    if status == "success":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "error":
        print(f"{RED}❌ {message}{RESET}")
    elif status == "warning":
        print(f"{YELLOW}⚠️  {message}{RESET}")
    else:
        print(f"{BLUE}ℹ️  {message}{RESET}")

def check_env_variables():
    """Check if required environment variables are set"""
    print("\n" + "="*50)
    print("Environment Variable Check")
    print("="*50)
    
    roles = {
        "Retriever": {
            "provider": os.environ.get("RETRIEVER_PROVIDER", "gemini"),
            "model": os.environ.get("RETRIEVER_MODEL", "gemini-2.0-flash-exp"),
            "api_key": os.environ.get("RETRIEVER_API_KEY") or os.environ.get("GEMINI_API_KEY")
        },
        "Global": {
            "provider": os.environ.get("GLOBAL_PROVIDER", "perplexity"),
            "model": os.environ.get("GLOBAL_MODEL", "llama-3.1-sonar-large-128k-online"),
            "api_key": os.environ.get("GLOBAL_API_KEY") or os.environ.get("PERPLEXITY_API_KEY")
        },
        "Synthesizer": {
            "provider": os.environ.get("SYNTH_PROVIDER", "perplexity"),
            "model": os.environ.get("SYNTH_MODEL", "llama-3.1-sonar-large-128k-online"),
            "api_key": os.environ.get("SYNTH_API_KEY") or os.environ.get("PERPLEXITY_API_KEY")
        },
        "Reasoner": {
            "provider": os.environ.get("REASON_PROVIDER", "anthropic"),
            "model": os.environ.get("REASON_MODEL", "claude-3-5-sonnet-20241022"),
            "api_key": os.environ.get("REASON_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        }
    }
    
    all_good = True
    for role, config in roles.items():
        print(f"\n{role}:")
        print(f"  Provider: {config['provider']}")
        print(f"  Model: {config['model']}")
        
        if config['api_key']:
            masked_key = config['api_key'][:10] + "..." + config['api_key'][-4:] if len(config['api_key']) > 14 else "***"
            print_status(f"API Key: {masked_key}", "success")
        else:
            if config['provider'] == "manual":
                print_status("API Key: Not needed (manual mode)", "warning")
            else:
                print_status("API Key: NOT SET", "error")
                all_good = False
    
    return all_good

def test_service_health(url: str, name: str) -> bool:
    """Test if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print_status(f"{name} is running at {url}", "success")
            return True
        else:
            print_status(f"{name} returned status {response.status_code}", "error")
            return False
    except requests.exceptions.ConnectionError:
        print_status(f"{name} is not responding at {url}", "error")
        return False
    except Exception as e:
        print_status(f"{name} error: {str(e)}", "error")
        return False

def test_services():
    """Test all services"""
    print("\n" + "="*50)
    print("Service Health Checks")
    print("="*50)
    
    services = [
        ("http://localhost:8000", "Gemini Retriever"),
        ("http://localhost:8001", "Orchestrator"),
        ("http://localhost:8002", "Embedding Service"),
    ]
    
    all_healthy = True
    for url, name in services:
        if not test_service_health(url, name):
            all_healthy = False
    
    return all_healthy

def test_retriever():
    """Test the retriever with a simple query"""
    print("\n" + "="*50)
    print("Retriever Test")
    print("="*50)
    
    url = "http://localhost:8000/gemini-flash-retrieve"
    payload = {
        "query": "What is authentication?",
        "shard_name": "test_shard",
        "shard_docs": [
            "Authentication is the process of verifying identity.",
            "Users log in with username and password.",
            "JWT tokens are used for session management."
        ]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print_status("Retriever responded successfully", "success")
            print(f"  Shard: {result.get('shard', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A')}")
            print(f"  Context items: {len(result.get('context', []))}")
            return True
        else:
            print_status(f"Retriever returned status {response.status_code}", "error")
            return False
    except Exception as e:
        print_status(f"Retriever test failed: {str(e)}", "error")
        return False

def test_minimal_pipeline():
    """Test the pipeline with a minimal query"""
    print("\n" + "="*50)
    print("Minimal Pipeline Test")
    print("="*50)
    
    url = "http://localhost:8001/query"
    payload = {
        "query": "What is a REST API?",
        "shards": []  # No shards for minimal test
    }
    
    try:
        print_status("Sending query to orchestrator...", "info")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print_status("Pipeline completed successfully", "success")
            
            # Display results
            if "master_prompt" in result:
                print(f"\nMaster Prompt (first 200 chars):")
                print(f"  {result['master_prompt'][:200]}...")
            
            if "answer" in result:
                print(f"\nAnswer (first 200 chars):")
                print(f"  {result['answer'][:200]}...")
            
            if "metadata" in result:
                print(f"\nMetadata:")
                print(f"  Model: {result['metadata'].get('model', 'N/A')}")
                print(f"  Answer ID: {result['metadata'].get('answer_id', 'N/A')[:8]}...")
            
            return True
        else:
            print_status(f"Pipeline returned status {response.status_code}", "error")
            if response.text:
                print(f"  Error: {response.text[:200]}")
            return False
    except Exception as e:
        print_status(f"Pipeline test failed: {str(e)}", "error")
        return False

def estimate_costs():
    """Estimate costs based on current configuration"""
    print("\n" + "="*50)
    print("Cost Estimation")
    print("="*50)
    
    reason_provider = os.environ.get("REASON_PROVIDER", "anthropic")
    reason_model = os.environ.get("REASON_MODEL", "claude-3-5-sonnet-20241022")
    
    print("Cost per query estimates:")
    print("  Minimal (Gemini only): ~$0.01-0.02")
    print("  Balanced (Mixed providers): ~$0.05-0.08")
    print("  Premium (Best models): ~$0.15-0.25")
    
    print(f"\nYour current configuration:")
    if reason_provider == "manual":
        print_status("~$0.02-0.04 (manual reasoning mode)", "success")
    elif "haiku" in reason_model.lower():
        print_status("~$0.05-0.08 (using Claude Haiku)", "success")
    elif "sonnet" in reason_model.lower():
        print_status("~$0.10-0.15 (using Claude Sonnet)", "warning")
    elif "opus" in reason_model.lower():
        print_status("~$0.20-0.30 (using Claude Opus)", "warning")
    else:
        print_status("~$0.05-0.10 (standard configuration)", "info")

def create_env_template():
    """Create a template .env file"""
    template = """# Agent-Based RAG System Configuration

# Retriever Role (Shard Search)
RETRIEVER_PROVIDER=gemini
RETRIEVER_MODEL=gemini-2.0-flash-exp
RETRIEVER_API_KEY=your-gemini-key-here

# Global Search Role
GLOBAL_PROVIDER=perplexity
GLOBAL_MODEL=llama-3.1-sonar-large-128k-online
GLOBAL_API_KEY=your-perplexity-key-here

# Synthesizer Role
SYNTH_PROVIDER=perplexity
SYNTH_MODEL=llama-3.1-sonar-large-128k-online
SYNTH_API_KEY=your-perplexity-key-here
SYNTH_TOKEN_LIMIT=4000

# Reasoner Role
REASON_PROVIDER=anthropic
REASON_MODEL=claude-3-5-sonnet-20241022
REASON_API_KEY=your-anthropic-key-here
REASON_MAX_TOKENS=2000

# Optional: Obsidian Integration
OBSIDIAN_VAULT=/path/to/your/vault

# Optional: Override modes
# USE_GEMINI_ONLY=true
# REASONING_MODE=manual
"""
    
    env_path = Path(".env.template")
    env_path.write_text(template)
    print_status(f"Created {env_path}", "success")
    print("  Edit this file with your API keys and rename to .env")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 Agent-Based RAG System - Test Suite")
    print("="*60)
    
    # Check what to test
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "env":
            check_env_variables()
        elif command == "services":
            test_services()
        elif command == "retriever":
            test_retriever()
        elif command == "pipeline":
            test_minimal_pipeline()
        elif command == "costs":
            estimate_costs()
        elif command == "template":
            create_env_template()
        elif command == "all":
            # Run all tests
            env_ok = check_env_variables()
            services_ok = test_services()
            
            if services_ok:
                retriever_ok = test_retriever()
                if retriever_ok:
                    test_minimal_pipeline()
            
            estimate_costs()
        else:
            print("Usage: python test_utilities.py [command]")
            print("\nCommands:")
            print("  env       - Check environment variables")
            print("  services  - Test service health")
            print("  retriever - Test retriever service")
            print("  pipeline  - Test full pipeline")
            print("  costs     - Estimate costs")
            print("  template  - Create .env template")
            print("  all       - Run all tests")
    else:
        # Default: check env and services
        check_env_variables()
        test_services()
        print("\n" + "="*60)
        print("Run 'python test_utilities.py all' for complete test suite")
        print("="*60)

if __name__ == "__main__":
    main()
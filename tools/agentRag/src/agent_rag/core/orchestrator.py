#!/usr/bin/env python3
"""
Agent-Based RAG Orchestrator
Main orchestration service for multi-agent retrieval and synthesis
"""

import os
import sys
from pathlib import Path

# Add parent directories to path for imports when running directly
if __name__ == "__main__":
    # When running as script, add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import json
import uuid
import sqlite3
import requests
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Import configuration
try:
    from agent_rag.config import get_config
    CONFIG = get_config()
except ImportError:
    # Fallback for direct execution
    from dotenv import load_dotenv
    load_dotenv()
    
    CONFIG = {
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

# Import provider extensions if available
try:
    from agent_rag.core.provider_extensions import get_provider, UniversalAgent
    EXTENDED_PROVIDERS = ["openai", "mistral", "groq", "together", 
                          "deepseek", "xai", "grok", "replicate"]
except ImportError:
    try:
        from provider_extensions import get_provider, UniversalAgent
        EXTENDED_PROVIDERS = ["openai", "mistral", "groq", "together", 
                              "deepseek", "xai", "grok", "replicate"]
    except ImportError:
        EXTENDED_PROVIDERS = []
        UniversalAgent = None


@dataclass
class PromptPackage:
    """Structured output from retrieval agents"""
    shard: str
    context: List[str]
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class QueryResult:
    """Final result from the orchestration pipeline"""
    query: str
    master_prompt: str
    final_answer: str
    artifact_path: Optional[str] = None
    metadata: Dict[str, Any] = None

class DatabaseManager:
    """Manages SQLite database for answer registry and embeddings"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Answers table
        c.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                answer_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                timestamp TEXT,
                model TEXT,
                prompt_id TEXT,
                vector BLOB,
                tags TEXT,
                uri TEXT,
                query TEXT,
                master_prompt TEXT
            )
        """)
        
        # Chunks table for future vector search
        c.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_macro_id TEXT,
                text TEXT NOT NULL,
                embedding BLOB,
                chunk_type TEXT,
                source_file TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_answer(self, answer_id: str, file_path: str, model: str, 
                   query: str, master_prompt: str, **kwargs):
        """Log an answer to the database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO answers (answer_id, file_path, timestamp, model, 
                               query, master_prompt, tags, uri)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            answer_id, 
            file_path, 
            datetime.now().isoformat(),
            model,
            query,
            master_prompt,
            kwargs.get('tags'),
            kwargs.get('uri')
        ))
        
        conn.commit()
        conn.close()

class ShardRetriever:
    """Handles shard-based retrieval using configured provider"""
    
    def __init__(self, role_config: Dict):
        self.api_url = role_config["endpoint"]
        self.provider = role_config["provider"]
        self.api_key = role_config["api_key"]
    
    async def retrieve(self, query: str, shards: List[Dict]) -> List[PromptPackage]:
        """Retrieve relevant chunks from multiple shards"""
        results = []
        
        # In production, use asyncio.gather for parallel requests
        for shard in shards:
            try:
                response = requests.post(self.api_url, json={
                    "query": query,
                    "shard_name": shard["shard_name"],
                    "shard_docs": shard["shard_docs"]
                })
                response.raise_for_status()
                data = response.json()
                
                package = PromptPackage(
                    shard=data["shard"],
                    context=data["context"],
                    confidence=data["confidence"]
                )
                results.append(package)
                
            except Exception as e:
                logger.error(f"Shard retrieval failed for {shard['shard_name']}: {e}")
                continue
        
        return results

class GlobalRAGAgent:
    """Handles global retrieval using configured provider"""
    
    def __init__(self, role_config: Dict):
        self.provider = role_config["provider"]
        self.model = role_config["model"]
        self.api_key = role_config["api_key"]
        self.api_url = role_config["endpoint"]
    
    async def search(self, query: str) -> PromptPackage:
        """Perform global search across all sources"""
        
        # Handle different providers
        if self.provider == "gemini":
            return await self._search_gemini(query)
        elif self.provider == "perplexity":
            return await self._search_perplexity(query)
        else:
            logger.error(f"Unknown global provider: {self.provider}")
            return PromptPackage(shard="global_rag", context=[], confidence=0)
    
    async def _search_perplexity(self, query: str) -> PromptPackage:
        """Search using Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are a global retrieval agent with access to the entire codebase and external sources.
        Return JSON with shard, context[], confidence.
        
        QUERY: {query}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a global retrieval agent. Return JSON."},
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            
            return PromptPackage(
                shard="global_rag",
                context=data.get("context", []),
                confidence=data.get("confidence", 0.5)
            )
        except Exception as e:
            logger.error(f"Global RAG failed: {e}")
            return PromptPackage(shard="global_rag", context=[], confidence=0)
    
    async def _search_gemini(self, query: str) -> PromptPackage:
        """Search using Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            prompt = f"""You are a global retrieval agent. Given query: {query}
            Return the 3 most relevant pieces of information as a concise list."""
            
            response = model.generate_content(prompt)
            text = response.text or ""
            
            # Parse into context list
            context = [s.strip() for s in text.split('\n') if s.strip()][:3]
            
            return PromptPackage(
                shard="global_rag",
                context=context,
                confidence=0.7
            )
        except Exception as e:
            logger.error(f"Gemini global search failed: {e}")
            return PromptPackage(shard="global_rag", context=[], confidence=0)

class SynthesisAgent:
    """Merges and compresses prompt packages using configured provider"""
    
    def __init__(self, role_config: Dict):
        self.provider = role_config["provider"]
        self.model = role_config["model"]
        self.api_key = role_config["api_key"]
        self.api_url = role_config["endpoint"]
        self.token_limit = role_config.get("token_limit", 4000)
    
    def get_manual_synthesis_prompt(self, packages: List[PromptPackage], 
                                   query: str, token_limit: int) -> str:
        """Generate instructions for manual synthesis"""
        packages_dict = [
            {
                "shard": p.shard,
                "context": p.context,
                "confidence": p.confidence
            } for p in packages
        ]
        
        instructions = f"""# Master Prompt Synthesis Instructions

You are helping to synthesize multiple retrieval outputs into a single, coherent Master Prompt that will be used for answering a question.

## Original Query
{query}

## Your Task
1. Review all the retrieved context packages below
2. Merge them into a single, coherent narrative
3. Remove any duplicate or redundant information
4. Preserve all critical technical details
5. Organize the information logically
6. Compress the output to fit within approximately {token_limit} tokens
7. Return ONLY the synthesized Master Prompt text (no explanations or metadata)

## Retrieved Context Packages

{json.dumps(packages_dict, indent=2)}

## Instructions for Synthesis
- Start with the most relevant and high-confidence contexts
- Group related information together
- Maintain technical accuracy
- Keep code snippets and examples intact
- Preserve important relationships between concepts
- Ensure the final output directly supports answering the original query

Please generate the Master Prompt now:"""
        
        return instructions
    
    async def synthesize(self, packages: List[PromptPackage], 
                        token_limit: Optional[int] = None,
                        query: Optional[str] = None) -> str:
        """Synthesize multiple prompt packages into a master prompt"""
        
        if token_limit is None:
            token_limit = self.token_limit
        
        # Handle manual synthesis mode
        if self.provider == "manual":
            if not query:
                query = "No specific query provided"
            return self.get_manual_synthesis_prompt(packages, query, token_limit)
        
        # Convert packages to dict for serialization
        packages_dict = [
            {
                "shard": p.shard,
                "context": p.context,
                "confidence": p.confidence
            } for p in packages
        ]
        
        # Route to appropriate provider
        if self.provider == "gemini":
            return await self._synthesize_gemini(packages_dict, token_limit)
        elif self.provider == "perplexity":
            return await self._synthesize_perplexity(packages_dict, token_limit)
        else:
            logger.error(f"Unknown synthesis provider: {self.provider}")
            # Fallback: concatenate contexts
            all_contexts = []
            for p in packages:
                all_contexts.extend(p.context)
            return "\n\n".join(all_contexts[:10])
    
    async def _synthesize_perplexity(self, packages_dict: List[Dict], 
                                     token_limit: int) -> str:
        """Synthesize using Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are a synthesis agent that merges multiple retrieval outputs into one coherent Master Prompt.
        
        TASK:
        1. Merge all contexts into a single narrative
        2. Remove duplicates
        3. Preserve critical technical details
        4. Compress to fit within {token_limit} tokens
        
        OUTPUT: Return only the merged context text (not JSON)
        
        PROMPT PACKAGES:
        {json.dumps(packages_dict, indent=2)}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You merge retrieval outputs. Return plain text."},
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback
            all_contexts = []
            for p in packages_dict:
                all_contexts.extend(p.get("context", []))
            return "\n\n".join(all_contexts[:10])
    
    async def _synthesize_gemini(self, packages_dict: List[Dict], 
                                 token_limit: int) -> str:
        """Synthesize using Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            prompt = f"""You merge multiple retrieval outputs into a coherent Master Prompt.
            Compress to fit within {token_limit} tokens while preserving all critical details.
            
            PROMPT PACKAGES:
            {json.dumps(packages_dict, indent=2)}
            
            Return the merged context as plain text."""
            
            response = model.generate_content(prompt)
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini synthesis failed: {e}")
            # Fallback
            all_contexts = []
            for p in packages_dict:
                all_contexts.extend(p.get("context", []))
            return "\n\n".join(all_contexts[:10])

class ReasoningAgent:
    """Final reasoning using configured provider"""
    
    def __init__(self, role_config: Dict):
        self.provider = role_config["provider"]
        self.model = role_config["model"]
        self.api_key = role_config["api_key"]
        self.api_url = role_config["endpoint"]
        self.max_tokens = role_config.get("max_tokens", 2000)
    
    async def reason(self, master_prompt: str, query: str) -> Dict[str, Any]:
        """Generate final answer using configured provider"""
        
        # Handle manual mode
        if self.provider == "manual":
            return {
                "reasoning_steps": ["Manual mode - paste Master Prompt to chat UI"],
                "final_output": "Copy the Master Prompt above and paste into your preferred chat interface (Claude, ChatGPT, etc.) for final reasoning.",
                "model": "manual"
            }
        
        # Route to appropriate provider
        if self.provider == "anthropic":
            return await self._reason_anthropic(master_prompt, query)
        elif self.provider == "gemini":
            return await self._reason_gemini(master_prompt, query)
        elif self.provider == "perplexity":
            return await self._reason_perplexity(master_prompt, query)
        else:
            logger.error(f"Unknown reasoning provider: {self.provider}")
            return {
                "reasoning_steps": ["Error: Unknown provider"],
                "final_output": f"Provider {self.provider} not supported",
                "model": "error"
            }
    
    async def _reason_anthropic(self, master_prompt: str, query: str) -> Dict[str, Any]:
        """Reason using Anthropic Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        system_prompt = """You are a reasoning and implementation agent with deep expertise.
        Based on the provided context, answer the user's question thoroughly and accurately.
        Include code examples where relevant."""
        
        user_prompt = f"""CONTEXT:
{master_prompt}

QUESTION:
{query}

Please provide a comprehensive answer based on the context above."""
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "system": system_prompt
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            content = response.json()["content"][0]["text"]
            
            return {
                "reasoning_steps": [],
                "final_output": content,
                "model": self.model
            }
        except Exception as e:
            logger.error(f"Anthropic reasoning failed: {e}")
            return {
                "reasoning_steps": ["Error occurred"],
                "final_output": f"Failed to generate answer: {str(e)}",
                "model": "error"
            }
    
    async def _reason_gemini(self, master_prompt: str, query: str) -> Dict[str, Any]:
        """Reason using Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            prompt = f"""Based on the following context, provide a comprehensive answer to the question.
            
CONTEXT:
{master_prompt}

QUESTION:
{query}

Answer thoroughly with examples where relevant."""
            
            response = model.generate_content(prompt)
            
            return {
                "reasoning_steps": [],
                "final_output": response.text or "",
                "model": self.model
            }
        except Exception as e:
            logger.error(f"Gemini reasoning failed: {e}")
            return {
                "reasoning_steps": ["Error occurred"],
                "final_output": f"Failed to generate answer: {str(e)}",
                "model": "error"
            }
    
    async def _reason_perplexity(self, master_prompt: str, query: str) -> Dict[str, Any]:
        """Reason using Perplexity API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""Based on the context provided, answer the question comprehensively.
        
CONTEXT:
{master_prompt}

QUESTION:
{query}"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides detailed answers based on context."},
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"]
            
            return {
                "reasoning_steps": [],
                "final_output": content,
                "model": self.model
            }
        except Exception as e:
            logger.error(f"Perplexity reasoning failed: {e}")
            return {
                "reasoning_steps": ["Error occurred"],
                "final_output": f"Failed to generate answer: {str(e)}",
                "model": "error"
            }

class Orchestrator:
    """Main orchestration engine"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db = DatabaseManager(config["db_path"])
        
        # Initialize agents with role-based configuration
        self.shard_retriever = ShardRetriever(config["roles"]["retriever"])
        self.global_rag = GlobalRAGAgent(config["roles"]["global"])
        self.synthesizer = SynthesisAgent(config["roles"]["synthesizer"])
        self.reasoner = ReasoningAgent(config["roles"]["reasoner"])
    
    async def process_query(self, query: str, shards: List[Dict] = None) -> QueryResult:
        """Main orchestration pipeline"""
        logger.info(f"Processing query: {query}")
        
        # Step 1: Shard Retrieval
        shard_results = []
        if shards:
            logger.info("Step 1: Shard Retrieval...")
            shard_results = await self.shard_retriever.retrieve(query, shards)
        
        # Step 2: Global RAG
        logger.info("Step 2: Global RAG...")
        global_context = await self.global_rag.search(query)
        
        # Step 3: Synthesis
        logger.info("Step 3: Synthesis...")
        all_packages = shard_results + [global_context]
        
        # Check if synthesis is manual mode
        if self.config["roles"]["synthesizer"]["provider"] == "manual":
            master_prompt = await self.synthesizer.synthesize(
                all_packages, 
                query=query  # Pass query for manual instructions
            )
            # In manual mode, master_prompt contains instructions
            # User needs to generate actual master prompt manually
            return QueryResult(
                query=query,
                master_prompt=master_prompt,
                final_answer="Manual synthesis mode: Copy the synthesis instructions above to generate the Master Prompt, then paste it back for final reasoning.",
                metadata={
                    "answer_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "synthesis_mode": "manual",
                    "model": "manual_synthesis"
                }
            )
        else:
            master_prompt = await self.synthesizer.synthesize(all_packages)
        
        # Step 4: Reasoning
        logger.info("Step 4: Reasoning...")
        final_answer = await self.reasoner.reason(master_prompt, query)
        
        # Log to database
        answer_id = str(uuid.uuid4())
        self.db.log_answer(
            answer_id=answer_id,
            file_path=None,  # Will be set by Obsidian integration
            model=final_answer["model"],
            query=query,
            master_prompt=master_prompt[:500]  # Store first 500 chars
        )
        
        return QueryResult(
            query=query,
            master_prompt=master_prompt,
            final_answer=final_answer["final_output"],
            metadata={
                "answer_id": answer_id,
                "timestamp": datetime.now().isoformat(),
                "model": final_answer["model"]
            }
        )

# FastAPI integration (optional)
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    
    app = FastAPI(title="Agent-Based RAG Orchestrator")
    
    class QueryRequest(BaseModel):
        query: str
        shards: Optional[List[Dict]] = None
        reasoning_mode: Optional[str] = None  # Override reasoning mode
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "orchestrator",
            "roles": {
                role: {
                    "provider": CONFIG["roles"][role]["provider"],
                    "model": CONFIG["roles"][role]["model"]
                } for role in CONFIG["roles"]
            }
        }
    
    @app.post("/query")
    async def process_query_endpoint(request: QueryRequest):
        """Process a query through the orchestration pipeline"""
        # Allow reasoning mode override
        config = CONFIG.copy()
        if request.reasoning_mode:
            config["reasoning_mode"] = request.reasoning_mode
            if request.reasoning_mode == "manual":
                config["roles"]["reasoner"]["provider"] = "manual"
        
        orchestrator = Orchestrator(config)
        result = await orchestrator.process_query(request.query, request.shards)
        
        return {
            "answer_id": result.metadata["answer_id"],
            "query": result.query,
            "answer": result.final_answer,
            "master_prompt": result.master_prompt[:500],  # Truncate for response
            "metadata": result.metadata
        }
    
except ImportError:
    # FastAPI not installed, skip API setup
    pass

# CLI interface
async def main():
    """Command-line interface for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py 'Your question here'")
        sys.exit(1)
    
    query = sys.argv[1]
    
    # Example shards (replace with actual content)
    shards = [
        {
            "shard_name": "auth_module",
            "shard_docs": ["Authentication uses JWT tokens", "Tokens expire after 24 hours"]
        },
        {
            "shard_name": "database",
            "shard_docs": ["Uses PostgreSQL for storage", "Connection pooling enabled"]
        }
    ]
    
    orchestrator = Orchestrator(CONFIG)
    result = await orchestrator.process_query(query, shards)
    
    print(f"\n📋 Query: {result.query}")
    print(f"\n📝 Master Prompt (truncated):\n{result.master_prompt[:500]}...")
    print(f"\n✅ Final Answer:\n{result.final_answer}")
    print(f"\n🔖 Metadata: {json.dumps(result.metadata, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
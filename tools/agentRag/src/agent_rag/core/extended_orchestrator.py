"""
Extended Orchestrator with Code Storage and Local Model Support
Integrates code retrieval storage and local models into the RAG pipeline
"""

import os
import sys
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add parent directories for imports
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Import base orchestrator
try:
    from agent_rag.core.orchestrator import (
        Orchestrator, QueryResult, PromptPackage, 
        ShardRetriever, GlobalRAGAgent, SynthesisAgent, ReasoningAgent
    )
    from agent_rag.config import get_config
except ImportError:
    # Fallback imports for direct execution
    from orchestrator import (
        Orchestrator, QueryResult, PromptPackage,
        ShardRetriever, GlobalRAGAgent, SynthesisAgent, ReasoningAgent
    )
    
# Import new modules
from code_storage import CodeRetrievalPipeline, CodePartitioner, CodeChunk
from local_model_integration import (
    LocalModelManager, LocalCodeRetriever, 
    LocalSynthesizer, LocalReasoner, get_local_model_config
)

logger = logging.getLogger(__name__)

class ExtendedShardRetriever(ShardRetriever):
    """Enhanced shard retriever with code storage support"""
    
    def __init__(self, role_config: Dict, code_pipeline: Optional[CodeRetrievalPipeline] = None,
                 local_retriever: Optional[LocalCodeRetriever] = None):
        super().__init__(role_config)
        self.code_pipeline = code_pipeline
        self.local_retriever = local_retriever
    
    async def retrieve_from_code_files(self, query: str, file_paths: List[str], 
                                       session_id: str) -> List[PromptPackage]:
        """Retrieve from code files with partitioning and storage"""
        packages = []
        
        for file_path in file_paths:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Partition file into chunks
            partitioner = CodePartitioner()
            chunks = partitioner.partition_file(content, file_path)
            
            # Use local model for retrieval if available
            if self.local_retriever:
                relevant_chunks = await self.local_retriever.retrieve(query, chunks, top_k=5)
            else:
                # Fallback to taking first N chunks or all if small
                relevant_chunks = chunks[:10] if len(chunks) > 10 else chunks
            
            # Store in database if pipeline available
            if self.code_pipeline:
                self.code_pipeline.store_retrieval(relevant_chunks, session_id, query)
            
            # Convert to PromptPackages
            for chunk in relevant_chunks:
                package = PromptPackage(
                    shard=f"code_{Path(file_path).stem}",
                    context=[chunk.content],
                    confidence=chunk.relevance if chunk.relevance > 0 else 0.5,
                    metadata={
                        'file_path': chunk.file_path,
                        'lines': f"{chunk.start_line}-{chunk.end_line}",
                        'type': chunk.chunk_type
                    }
                )
                packages.append(package)
        
        return packages

class ExtendedSynthesisAgent(SynthesisAgent):
    """Enhanced synthesis with code deduplication"""
    
    def __init__(self, role_config: Dict, code_pipeline: Optional[CodeRetrievalPipeline] = None,
                 local_synthesizer: Optional[LocalSynthesizer] = None):
        super().__init__(role_config)
        self.code_pipeline = code_pipeline
        self.local_synthesizer = local_synthesizer
    
    async def synthesize_with_code(self, packages: List[PromptPackage], 
                                   session_id: Optional[str] = None,
                                   query: Optional[str] = None) -> str:
        """Synthesize with code deduplication support"""
        
        # If we have a session with stored code, prepare from database
        if session_id and self.code_pipeline:
            stored_content = self.code_pipeline.prepare_for_synthesis(
                session_id, 
                max_tokens=self.token_limit
            )
            
            # If local synthesizer available, use it
            if self.local_synthesizer and query:
                return await self.local_synthesizer.synthesize(stored_content, query)
            
            # Otherwise use base synthesis on the prepared content
            if stored_content:
                # Create a package from stored content
                stored_package = PromptPackage(
                    shard="stored_code",
                    context=[stored_content],
                    confidence=1.0
                )
                packages = [stored_package] + packages
        
        # Use base synthesis
        return await self.synthesize(packages, query=query)

class ExtendedReasoningAgent(ReasoningAgent):
    """Enhanced reasoning with local model support"""
    
    def __init__(self, role_config: Dict, local_reasoner: Optional[LocalReasoner] = None):
        super().__init__(role_config)
        self.local_reasoner = local_reasoner
    
    async def reason_with_fallback(self, master_prompt: str, query: str) -> Dict[str, Any]:
        """Reason with local model fallback"""
        
        # Try local model first if configured for cost savings
        if self.local_reasoner and self.provider == "local":
            try:
                response = await self.local_reasoner.reason(master_prompt, query)
                return {
                    "reasoning_steps": ["Local model reasoning"],
                    "final_output": response,
                    "model": "local"
                }
            except Exception as e:
                logger.warning(f"Local reasoning failed, falling back: {e}")
        
        # Use base reasoning
        return await self.reason(master_prompt, query)

class ExtendedOrchestrator(Orchestrator):
    """Extended orchestrator with code storage, local model support, and optional hybrid refinement."""

    def __init__(self, config: Dict):
        # Initialize base orchestrator
        super().__init__(config)

        # Initialize code storage if enabled (unchanged)
        self.code_pipeline = None
        if config.get("code_retrieval", {}).get("enable_storage", False):
            storage_config = config.get("code_retrieval", {})
            self.code_pipeline = CodeRetrievalPipeline(storage_config)

        # Initialize local models if enabled (unchanged)
        self.model_manager = None
        self.local_retriever = None
        self.local_synthesizer = None
        self.local_reasoner = None

        local_config = get_local_model_config()
        if local_config.get("enabled", False):
            self.model_manager = LocalModelManager(local_config.get("models_dir", "./models"))
            self.local_retriever = LocalCodeRetriever(self.model_manager)
            self.local_synthesizer = LocalSynthesizer(self.model_manager)
            self.local_reasoner = LocalReasoner(self.model_manager)

        # Replace agents with extended versions (unchanged)
        self.shard_retriever = ExtendedShardRetriever(
            config["roles"]["retriever"],
            self.code_pipeline,
            self.local_retriever
        )

        self.synthesizer = ExtendedSynthesisAgent(
            config["roles"]["synthesizer"],
            self.code_pipeline,
            self.local_synthesizer
        )

        self.reasoner = ExtendedReasoningAgent(
            config["roles"]["reasoner"],
            self.local_reasoner
        )

        # -------------------------
        # Additive hybrid retrieval
        # -------------------------
        self.logger = logging.getLogger(__name__)
        retrieval_cfg = (config or {}).get("retrieval", {})
        self.long_context_mode: bool = bool(retrieval_cfg.get("long_context_mode", False))
        self.top_m_files_for_refine: int = int(retrieval_cfg.get("top_m_files_for_refine", 5))
        self.preview_chars: int = int(retrieval_cfg.get("preview_chars", 200))

        # Optional Phase 2 modules; safe if absent
        self._select_top_files = None
        self._agent_refiner = None
        try:
            from file_selector import select_top_files as _sel
            self._select_top_files = _sel
        except Exception as e:
            self.logger.info(f"file_selector not available: {e}")
        try:
            from agent_refiner import AgentRefiner
            self._agent_refiner = AgentRefiner()
        except Exception as e:
            self.logger.info(f"agent_refiner not available: {e}")

    async def process_code_query(self, query: str, code_files: List[str],
                                 shards: Optional[List[Dict]] = None) -> QueryResult:
        """Process a query with code files (original flow remains intact)"""
        logger.info(f"Processing code query: {query}")

        session_id = str(uuid.uuid4())

        # Step 1: Code Retrieval
        code_packages = []
        if code_files:
            logger.info(f"Retrieving from {len(code_files)} code files...")
            code_packages = await self.shard_retriever.retrieve_from_code_files(
                query, code_files, session_id
            )

        # Step 2: Regular Shard Retrieval (if provided)
        shard_results = []
        if shards:
            logger.info("Retrieving from provided shards...")
            shard_results = await self.shard_retriever.retrieve(query, shards)

        # Step 3: Global RAG
        logger.info("Global RAG search...")
        global_context = await self.global_rag.search(query)

        # Step 4: Synthesis with deduplication
        logger.info("Synthesizing results...")
        all_packages = code_packages + shard_results + [global_context]

        # Use extended synthesis if we have code storage
        if isinstance(self.synthesizer, ExtendedSynthesisAgent):
            master_prompt = await self.synthesizer.synthesize_with_code(
                all_packages,
                session_id=session_id,
                query=query
            )
        else:
            master_prompt = await self.synthesizer.synthesize(all_packages, query=query)

        # Handle manual synthesis mode
        if self.config["roles"]["synthesizer"]["provider"] == "manual":
            return QueryResult(
                query=query,
                master_prompt=master_prompt,
                final_answer="Manual synthesis mode: Follow the instructions above.",
                metadata={
                    "answer_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "synthesis_mode": "manual",
                    "model": "manual_synthesis",
                    "session_id": session_id
                }
            )

        # Step 5: Reasoning with local model support
        logger.info("Generating final answer...")
        if isinstance(self.reasoner, ExtendedReasoningAgent):
            final_answer = await self.reasoner.reason_with_fallback(master_prompt, query)
        else:
            final_answer = await self.reasoner.reason(master_prompt, query)

        # Store annotations if any
        if self.code_pipeline and final_answer.get("annotations"):
            for annotation in final_answer["annotations"]:
                self.code_pipeline.store_annotation(
                    content_hash=annotation["hash"],
                    annotation_type=annotation["type"],
                    annotation_text=annotation["text"] if "text" in annotation else annotation.get("annotation_text", ""),
                    source_model=final_answer["model"],
                    confidence=annotation.get("confidence", 0.5),
                    relative_position=annotation.get("position")
                )

        # Log to database
        answer_id = str(uuid.uuid4())
        self.db.log_answer(
            answer_id=answer_id,
            file_path=None,
            model=final_answer["model"],
            query=query,
            master_prompt=master_prompt[:500]
        )

        return QueryResult(
            query=query,
            master_prompt=master_prompt,
            final_answer=final_answer["final_output"],
            metadata={
                "answer_id": answer_id,
                "timestamp": datetime.now().isoformat(),
                "model": final_answer["model"],
                "session_id": session_id
            }
        )

    # ---------------------------
    # Hybrid retrieval entrypoints
    # ---------------------------

    async def hybrid_retrieve_stage1(self, query: str, files: List[str]) -> List[Dict[str, Any]]:
        """
        Stage 1: heuristic shortlist using the existing retriever pipeline.
        Tries to pass preview_chars if supported; silently falls back if not.
        """
        if hasattr(self.shard_retriever, "retrieve_all"):
            try:
                return await self.shard_retriever.retrieve_all(
                    query=query,
                    files=files,
                    preview_chars=getattr(self, "preview_chars", 200),
                )
            except TypeError:
                return await self.shard_retriever.retrieve_all(query=query, files=files)
        # Fallback to base retrieve path if present
        if hasattr(super(), "process_query"):
            shards = [{"shard_name": Path(f).name, "shard_docs": [Path(f).read_text(encoding="utf-8")]} for f in files]
            result = await super().process_query(query, shards)
            return [{"file": None, "score": 0.0, "content_preview": result.master_prompt[:self.preview_chars]}]
        self.logger.warning("No available Stage 1 retriever; returning empty result set.")
        return []

    async def hybrid_refine_stage2(
        self,
        query: str,
        stage1_results: List[Dict[str, Any]],
        macro_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: optional long-context refinement for top-M files.
        Requires file_selector.select_top_files and agent_refiner.AgentRefiner; otherwise returns [].
        """
        if not self.long_context_mode:
            return []
        if not self._select_top_files or not self._agent_refiner:
            return []
        top_files = self._select_top_files(stage1_results, top_m=self.top_m_files_for_refine)
        if not top_files:
            return []
        try:
            refined = await self._agent_refiner.refine_files(
                query=query,
                files=top_files,
                macro_index=macro_index,
            )
            return refined or []
        except Exception as e:
            self.logger.info(f"Refine skipped due to error: {e}")
            return []

    def _hybrid_merge_and_dedupe(self, stage1: List[Dict[str, Any]], stage2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge heuristic and refined candidates; prefer higher score on conflicts.
        Keys normalize across {file|file_path} and {span|position}.
        """
        by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def k(item: Dict[str, Any]) -> Tuple[str, str]:
            file_path = item.get("file") or item.get("file_path") or ""
            span = item.get("span") or item.get("position") or ""
            return (file_path, str(span))

        for src in (stage1 or [], stage2 or []):
            for it in src:
                key = k(it)
                prev = by_key.get(key)
                if not prev or float(it.get("score", 0.0)) > float(prev.get("score", 0.0)):
                    by_key[key] = it

        return sorted(by_key.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)

    async def hybrid_retrieve_and_refine(
        self,
        query: str,
        files: List[str],
        macro_index: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        End-to-end hybrid path: Stage 1 always runs; Stage 2 runs only if enabled; returns merged list.
        """
        stage1 = await self.hybrid_retrieve_stage1(query, files)
        if not self.long_context_mode:
            return stage1
        stage2 = await self.hybrid_refine_stage2(query, stage1, macro_index=macro_index)
        return self._hybrid_merge_and_dedupe(stage1, stage2)

    # ----------------
    # Housekeeping API
    # ----------------

    def cleanup(self):
        """Clean up resources (unchanged)"""
        if self.code_pipeline:
            self.code_pipeline.cleanup_expired_sessions()
        if self.model_manager:
            self.model_manager.cleanup()

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics (unchanged)"""
        stats: Dict[str, Any] = {}
        if self.code_pipeline:
            db_size = self.code_pipeline.get_storage_size()
            stats["code_storage_size_mb"] = db_size / (1024 * 1024)
            cursor = self.code_pipeline.db.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM code_snippets")
            stats["total_snippets"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM code_metadata")
            stats["total_locations"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM code_annotations")
            stats["total_annotations"] = cursor.fetchone()[0]
        if self.model_manager:
            stats["available_local_models"] = self.model_manager.get_available_models()
        return stats


# Enhanced configuration
def get_extended_config() -> Dict[str, Any]:
    """Get configuration with code storage and local model settings"""
    base_config = get_config() if 'get_config' in globals() else {}
    
    # Add code retrieval configuration
    base_config["code_retrieval"] = {
        "enable_storage": os.getenv("ENABLE_CODE_STORAGE", "true").lower() == "true",
        "db_path": os.getenv("CODE_STORAGE_DB", "code_storage.db"),
        "dedup_threshold": float(os.getenv("DEDUP_THRESHOLD", "0.95")),
        "partition_strategy": os.getenv("PARTITION_STRATEGY", "ast"),  # ast | heuristic | sliding_window
        "max_chunk_tokens": int(os.getenv("MAX_CHUNK_TOKENS", "4000")),
        "chunk_overlap": float(os.getenv("CHUNK_OVERLAP", "0.1")),
        "session_ttl_hours": int(os.getenv("SESSION_TTL_HOURS", "24")),
        "max_storage_gb": int(os.getenv("MAX_STORAGE_GB", "10")),
        "synthesizer_context": int(os.getenv("SYNTHESIZER_CONTEXT", "10000"))
    }
    
    # Add local model configuration
    local_config = get_local_model_config()
    base_config["local_models"] = local_config
    
    # Override providers if using local models
    if local_config.get("enabled") and os.getenv("USE_LOCAL_FOR_ALL", "false").lower() == "true":
        base_config["roles"]["retriever"]["provider"] = "local"
        base_config["roles"]["synthesizer"]["provider"] = "local"
        base_config["roles"]["reasoner"]["provider"] = "local"
    
    return base_config

# FastAPI integration
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="Extended Agent-Based RAG Orchestrator")
    
    class CodeQueryRequest(BaseModel):
        query: str
        code_files: List[str]
        shards: Optional[List[Dict]] = None
    
    class StorageStatsResponse(BaseModel):
        code_storage_size_mb: float
        total_snippets: int
        total_locations: int
        total_annotations: int
        available_local_models: List[str]
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        config = get_extended_config()
        return {
            "status": "healthy",
            "service": "extended_orchestrator",
            "code_storage_enabled": config["code_retrieval"]["enable_storage"],
            "local_models_enabled": config["local_models"]["enabled"]
        }
    
    @app.post("/code_query")
    async def process_code_query_endpoint(request: CodeQueryRequest):
        """Process a code query"""
        config = get_extended_config()
        orchestrator = ExtendedOrchestrator(config)
        
        try:
            result = await orchestrator.process_code_query(
                request.query, 
                request.code_files,
                request.shards
            )
            
            return {
                "answer_id": result.metadata["answer_id"],
                "query": result.query,
                "answer": result.final_answer,
                "session_id": result.metadata.get("session_id"),
                "metadata": result.metadata
            }
        finally:
            orchestrator.cleanup()
    
    @app.get("/storage_stats", response_model=StorageStatsResponse)
    async def get_storage_stats():
        """Get storage statistics"""
        config = get_extended_config()
        orchestrator = ExtendedOrchestrator(config)
        stats = orchestrator.get_storage_stats()
        orchestrator.cleanup()
        return stats
    
    @app.post("/cleanup")
    async def cleanup_expired():
        """Clean up expired sessions"""
        config = get_extended_config()
        orchestrator = ExtendedOrchestrator(config)
        orchestrator.cleanup()
        return {"status": "cleaned"}
    
except ImportError:
    # FastAPI not installed
    pass

# CLI interface
async def main():
    """Command-line interface for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended RAG Orchestrator")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--code-files", nargs="+", help="Code files to process")
    parser.add_argument("--use-local", action="store_true", help="Use local models")
    parser.add_argument("--stats", action="store_true", help="Show storage stats")
    
    args = parser.parse_args()
    
    # Configure
    if args.use_local:
        os.environ["USE_LOCAL_MODELS"] = "true"
    
    config = get_extended_config()
    orchestrator = ExtendedOrchestrator(config)
    
    if args.stats:
        stats = orchestrator.get_storage_stats()
        print("\n📊 Storage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        # Process query
        result = await orchestrator.process_code_query(
            args.query,
            args.code_files or [],
            None
        )
        
        print(f"\n📋 Query: {result.query}")
        print(f"\n📄 Session ID: {result.metadata.get('session_id')}")
        print(f"\n✅ Answer:\n{result.final_answer}")
    
    orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
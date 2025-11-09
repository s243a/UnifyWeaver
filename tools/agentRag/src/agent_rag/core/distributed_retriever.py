# src/agent_rag/core/distributed_retriever.py

import os
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class ChunkConfig:
    max_chunk_size: int = 2048  # tokens (rough estimate: chars/4)
    overlap_size: int = 256     # tokens overlap between chunks
    overlap_ratio: float = 0.1  # alternative: overlap as ratio of chunk size
    
@dataclass
class FileChunk:
    file_path: str
    chunk_index: int
    content: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]

class DistributedRetriever:
    """Distributes file retrieval across multiple model instances"""
    
    def __init__(self, 
                 model_manager,
                 chunk_config: Optional[ChunkConfig] = None,
                 max_workers: int = 4,
                 preview_chars: int = 200
                 ):
        """
        Args:
            model_manager: LocalModelManager instance
            chunk_config: Configuration for chunking
            max_workers: Max parallel model queries
        """
        self.model_manager = model_manager
        self.config = chunk_config or ChunkConfig()
        self.max_workers = max_workers
        self.preview_chars = preview_chars
        self.chunk_cache = {}  # Cache processed chunks
        
    def retrieve(self, 
                 query: str,
                 file_paths: List[str],
                 top_k: int = 5,
                 score_threshold: float = 0.3) -> List[Dict]:
        """
        Main retrieval method that partitions files and queries models
        
        Args:
            query: The search query
            file_paths: List of file paths to search
            top_k: Number of top results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of relevant chunks with scores and metadata
        """
        # Step 1: Load and chunk all files
        all_chunks = self._prepare_chunks(file_paths)
        
        print(f"Prepared {len(all_chunks)} chunks from {len(file_paths)} files")
        
        # Step 2: Partition chunks across available models
        partitions = self._partition_chunks(all_chunks)
        
        # Step 3: Score chunks in parallel
        scored_chunks = self._score_chunks_parallel(query, partitions)
        
        # Step 4: Filter and sort results
        relevant = [
            chunk for chunk in scored_chunks 
            if chunk['score'] >= score_threshold
        ]
        relevant.sort(key=lambda x: x['score'], reverse=True)
        
        return relevant[:top_k]
    
    def _prepare_chunks(self, file_paths: List[str]) -> List[FileChunk]:
        """Load files and create chunks with overlap"""
        chunks = []
        
        for file_path in file_paths:
            # Check cache first
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{file_path}:{file_hash}"
            
            if cache_key in self.chunk_cache:
                chunks.extend(self.chunk_cache[cache_key])
                continue
            
            # Load and chunk file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_chunks = self._chunk_text(content, file_path)
                self.chunk_cache[cache_key] = file_chunks
                chunks.extend(file_chunks)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
        return chunks
    
    def _chunk_text(self, text: str, file_path: str) -> List[FileChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Rough token estimation (adjust based on your tokenizer)
        char_per_token = 4
        chunk_size_chars = self.config.max_chunk_size * char_per_token
        overlap_chars = int(self.config.overlap_size * char_per_token)
        
        # Handle overlap_ratio if specified
        if self.config.overlap_ratio > 0:
            overlap_chars = int(chunk_size_chars * self.config.overlap_ratio)
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            
            # Try to break at sentence/paragraph boundaries
            if end < len(text):
                # Look for sentence end
                for delim in ['\n\n', '. ', '.\n', '! ', '? ']:
                    delim_pos = text.rfind(delim, start, end)
                    if delim_pos > start + chunk_size_chars // 2:
                        end = delim_pos + len(delim)
                        break
            
            chunk_content = text[start:end]
            
            chunks.append(FileChunk(
                file_path=file_path,
                chunk_index=chunk_idx,
                content=chunk_content,
                start_pos=start,
                end_pos=end,
                metadata={
                    'file_name': os.path.basename(file_path),
                    'chunk_size': len(chunk_content),
                    'total_chunks': -1  # Will update after
                }
            ))
            
            chunk_idx += 1
            start = end - overlap_chars if end < len(text) else end
        
        # Update total chunks count
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
            
        return chunks
    
    def _partition_chunks(self, chunks: List[FileChunk]) -> List[List[FileChunk]]:
        """
        Partition chunks across available models.
        For now, simple round-robin. Could be smarter based on:
        - Model capacity/speed
        - Chunk complexity
        - File boundaries
        """
        # Determine number of partitions (could query multiple model instances)
        num_partitions = min(self.max_workers, len(chunks))
        partitions = [[] for _ in range(num_partitions)]
        
        for i, chunk in enumerate(chunks):
            partition_idx = i % num_partitions
            partitions[partition_idx].append(chunk)
            
        return partitions
    
    def _score_chunks_parallel(self, 
                               query: str, 
                               partitions: List[List[FileChunk]]) -> List[Dict]:
        """Score chunks in parallel across model instances"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit scoring tasks
            future_to_partition = {
                executor.submit(self._score_partition, query, partition): i
                for i, partition in enumerate(partitions)
            }
            
            # Collect results
            for future in as_completed(future_to_partition):
                partition_idx = future_to_partition[future]
                try:
                    results = future.result(timeout=30)
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error scoring partition {partition_idx}: {e}")
                    
        return all_results
    

    def _score_partition(self, query: str, partition: List[FileChunk]) -> List[Dict]:
        """
        Score a partition of chunks and emit results with numeric scores and
        a configurable preview length.
        """
        results: List[Dict] = []
        if not partition:
            return results

        chunk_texts = [chunk.content for chunk in partition]
        pairs = self.model_manager.score_chunks(query, chunk_texts)

        # Normalize to float scores; build map text -> score (max for duplicates)
        score_by_text: Dict[str, float] = {}
        for text, s in pairs:
            try:
                sv = float(s)
            except (TypeError, ValueError):
                if isinstance(s, (tuple, list)) and s:
                    try:
                        sv = float(s[0])
                    except (TypeError, ValueError):
                        sv = 0.0
                else:
                    sv = 0.0
            prev = score_by_text.get(text)
            score_by_text[text] = sv if prev is None else max(prev, sv)

        total_chunks = len(partition)
        for chunk in partition:
            s = float(score_by_text.get(chunk.content, 0.0))
            text = chunk.content
            limit = getattr(self, "preview_chars", 200)
            preview = text[:limit] + ("..." if len(text) > limit else "")
            position_str = f"{chunk.chunk_index}/{chunk.metadata.get('total_chunks', total_chunks)}" \
                           if isinstance(chunk.metadata, dict) else f"{chunk.chunk_index}/{total_chunks}"
            results.append({
                "chunk": chunk,
                "score": s,
                "file": chunk.file_path,
                "position": position_str,
                "content_preview": preview,
            })
        return results

    
    def _get_file_hash(self, file_path: str) -> str:
        """Get file hash for caching"""
        try:
            stat = os.stat(file_path)
            # Use mtime and size for quick hash
            return f"{stat.st_mtime}_{stat.st_size}"
        except:
            return "unknown"

# Test harness
def test_distributed_retriever():
    """Test the retriever with sample files"""
    import sys
    sys.path.append('src/agent_rag/core')
    from local_model_integration import LocalModelManager
    
    # Setup
    config = {
        'local_model_provider': 'ollama',
        'server_url': 'http://localhost:11434',
        'local_retrieval_model': 'phi3'
    }
    manager = LocalModelManager(config)
    
    # Configure chunking
    chunk_config = ChunkConfig(
        max_chunk_size=512,   # Smaller chunks for testing
        overlap_size=64
    )
    
    retriever = DistributedRetriever(
        model_manager=manager,
        chunk_config=chunk_config,
        max_workers=2
    )
    
    # Create test files
    test_dir = "test_results/retriever_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Sample content
    test_files = []
    
    # File 1: Authentication doc
    with open(f"{test_dir}/auth_doc.txt", "w") as f:
        f.write("""
Authentication and Authorization Guide

Authentication verifies the identity of a user or system. Common methods include:
- Username and password combinations
- Multi-factor authentication (MFA) using SMS or authenticator apps
- Biometric authentication using fingerprints or facial recognition
- Certificate-based authentication for systems

Authorization determines what an authenticated user can access. Key concepts:
- Role-based access control (RBAC) assigns permissions based on job functions
- Attribute-based access control (ABAC) uses user attributes
- Access control lists (ACLs) define specific resource permissions

JWT tokens are commonly used for stateless authentication in modern web applications.
They contain encoded claims about the user and are signed to prevent tampering.
        """)
        test_files.append(f"{test_dir}/auth_doc.txt")
    
    # File 2: Larger technical doc
    with open(f"{test_dir}/tech_guide.txt", "w") as f:
        f.write("""
Comprehensive System Architecture Guide

Chapter 1: Microservices Architecture
Microservices break applications into small, independent services that communicate via APIs.
Each service handles a specific business capability and can be developed, deployed, and scaled independently.
Benefits include improved fault isolation, technology diversity, and easier scaling.
Challenges include network complexity, data consistency, and service discovery.

Chapter 2: Database Design
Database normalization reduces redundancy and improves data integrity.
First Normal Form (1NF) eliminates repeating groups.
Second Normal Form (2NF) removes partial dependencies.
Third Normal Form (3NF) eliminates transitive dependencies.

Chapter 3: Security Best Practices
Security should be implemented in layers (defense in depth).
Input validation prevents injection attacks.
Encryption protects data at rest and in transit.
Regular security audits identify vulnerabilities.
Principle of least privilege limits potential damage from breaches.

Chapter 4: Performance Optimization
Caching reduces database load and improves response times.
Load balancing distributes traffic across multiple servers.
Database indexing speeds up query performance.
Code profiling identifies performance bottlenecks.
        """ * 3)  # Make it larger for chunking test
        test_files.append(f"{test_dir}/tech_guide.txt")
    
    # Run retrieval test
    queries = [
        "How does authentication work?",
        "What are microservices?",
        "Database normalization rules",
        "Security best practices"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        start_time = time.time()
        results = retriever.retrieve(
            query=query,
            file_paths=test_files,
            top_k=3,
            score_threshold=0.2
        )
        elapsed = time.time() - start_time
        
        print(f"\nFound {len(results)} relevant chunks in {elapsed:.2f}s")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"File: {os.path.basename(result['file'])}")
            print(f"Position: Chunk {result['position']}")
            print(f"Score: {result['score']:.3f}")
            print(f"Preview: {result['content_preview']}")
    
    # Save results
    results_file = f"{test_dir}/retrieval_results.json"
    with open(results_file, "w") as f:
        json.dump({
            'test_time': time.strftime("%Y%m%d_%H%M%S"),
            'config': {
                'chunk_size': chunk_config.max_chunk_size,
                'overlap': chunk_config.overlap_size
            },
            'queries': queries,
            'num_files': len(test_files)
        }, f, indent=2)
    
    print(f"\n\nResults saved to: {results_file}")

if __name__ == "__main__":
    test_distributed_retriever()
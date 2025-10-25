# src/agent_rag/core/code_aware_retriever.py

import os
import json
import ast
import re
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def _make_preview(text: str, limit: int) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")

class DocumentType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    RDF_TURTLE = "turtle"
    RDF_NTRIPLES = "ntriples"
    RDF_JSONLD = "jsonld"
    GENERIC_CODE = "code"
    TEXT = "text"

@dataclass
class ChunkConfig:
    max_chunk_size: int = 2048  # tokens
    overlap_size: int = 256     # tokens for text
    preserve_structure: bool = True  # Keep code structures intact
    include_context: bool = True     # Include imports/class defs in chunks
    
@dataclass
class CodeChunk:
    file_path: str
    chunk_index: int
    content: str
    chunk_type: str  # 'function', 'class', 'module', 'triple_group', etc
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    context: str  # Imports, class definition, namespace prefixes

class CodeAwareRetriever:
    def __init__(self, model_manager, chunk_config: Optional[ChunkConfig] = None, preview_chars: int = 200):
        self.model_manager = model_manager
        self.config = chunk_config or ChunkConfig()
        self.preview_chars = preview_chars
        self.max_workers = max_workers
        self.chunk_cache = {}
        
    def retrieve(self, 
                 query: str,
                 file_paths: List[str],
                 top_k: int = 5,
                 score_threshold: float = 0.3) -> List[Dict]:
        """Main retrieval with document type detection"""
        
        # Group files by type
        files_by_type = self._classify_files(file_paths)
        
        all_chunks = []
        for doc_type, paths in files_by_type.items():
            chunks = self._prepare_chunks_by_type(paths, doc_type)
            all_chunks.extend(chunks)
        
        print(f"Prepared {len(all_chunks)} chunks from {len(file_paths)} files")
        
        # Score chunks
        partitions = self._partition_chunks(all_chunks)
        scored_chunks = self._score_chunks_parallel(query, partitions)
        
        # Filter and sort
        relevant = [
            chunk for chunk in scored_chunks 
            if chunk['score'] >= score_threshold
        ]
        relevant.sort(key=lambda x: x['score'], reverse=True)
        
        return relevant[:top_k]
    
    def _classify_files(self, file_paths: List[str]) -> Dict[DocumentType, List[str]]:
        """Classify files by type based on extension and content"""
        files_by_type = {}
        
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            
            # Map extensions to document types
            if ext in ['.py']:
                doc_type = DocumentType.PYTHON
            elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                doc_type = DocumentType.JAVASCRIPT
            elif ext in ['.java']:
                doc_type = DocumentType.JAVA
            elif ext in ['.cpp', '.cc', '.h', '.hpp']:
                doc_type = DocumentType.CPP
            elif ext in ['.ttl', '.turtle']:
                doc_type = DocumentType.RDF_TURTLE
            elif ext in ['.nt', '.ntriples']:
                doc_type = DocumentType.RDF_NTRIPLES
            elif ext in ['.jsonld']:
                doc_type = DocumentType.RDF_JSONLD
            elif ext in ['.c', '.cs', '.go', '.rs', '.rb', '.php']:
                doc_type = DocumentType.GENERIC_CODE
            else:
                doc_type = DocumentType.TEXT
            
            if doc_type not in files_by_type:
                files_by_type[doc_type] = []
            files_by_type[doc_type].append(path)
            
        return files_by_type
    
    def _prepare_chunks_by_type(self, 
                                file_paths: List[str], 
                                doc_type: DocumentType) -> List[CodeChunk]:
        """Route to appropriate chunking strategy"""
        chunks = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if doc_type == DocumentType.PYTHON:
                    file_chunks = self._chunk_python(content, file_path)
                elif doc_type in [DocumentType.RDF_TURTLE, DocumentType.RDF_NTRIPLES]:
                    file_chunks = self._chunk_rdf(content, file_path, doc_type)
                elif doc_type in [DocumentType.JAVASCRIPT, DocumentType.JAVA, 
                                 DocumentType.CPP, DocumentType.GENERIC_CODE]:
                    file_chunks = self._chunk_generic_code(content, file_path)
                else:
                    file_chunks = self._chunk_text_fallback(content, file_path)
                
                chunks.extend(file_chunks)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        return chunks
    
    def _chunk_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk Python code by logical units (functions, classes)"""
        chunks = []
        lines = content.split('\n')
        
        # Extract imports and module-level context
        imports = []
        module_docs = []
        for i, line in enumerate(lines):
            if line.startswith(('import ', 'from ')):
                imports.append(line)
            elif i < 20 and '"""' in line:  # Capture module docstring
                module_docs.append(line)
        
        context = '\n'.join(imports)
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                chunk_content = None
                chunk_type = None
                start_line = None
                end_line = None
                
                if isinstance(node, ast.FunctionDef):
                    chunk_type = "function"
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    
                    # Include decorators
                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'lineno'):
                            start_line = min(start_line, decorator.lineno - 1)
                    
                    chunk_content = '\n'.join(lines[start_line:end_line])
                    
                elif isinstance(node, ast.ClassDef):
                    chunk_type = "class"
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    
                    # Include decorators
                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'lineno'):
                            start_line = min(start_line, decorator.lineno - 1)
                    
                    chunk_content = '\n'.join(lines[start_line:end_line])
                    
                    # Also chunk individual methods if class is large
                    class_lines = end_line - start_line
                    if class_lines > 50:  # Arbitrary threshold
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_chunk = CodeChunk(
                                    file_path=file_path,
                                    chunk_index=len(chunks),
                                    content=ast.unparse(item) if hasattr(ast, 'unparse') else self._get_source_segment(lines, item),
                                    chunk_type="method",
                                    start_line=item.lineno,
                                    end_line=item.end_lineno or item.lineno,
                                    metadata={
                                        'class_name': node.name,
                                        'method_name': item.name,
                                        'file_name': os.path.basename(file_path)
                                    },
                                    context=context + f"\nclass {node.name}:"
                                )
                                chunks.append(method_chunk)
                
                if chunk_content and chunk_type:
                    chunk = CodeChunk(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        content=chunk_content,
                        chunk_type=chunk_type,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            'name': node.name if hasattr(node, 'name') else 'unknown',
                            'file_name': os.path.basename(file_path)
                        },
                        context=context
                    )
                    chunks.append(chunk)
                    
        except SyntaxError as e:
            print(f"Syntax error in {file_path}, falling back to line-based chunking: {e}")
            chunks = self._chunk_generic_code(content, file_path)
            
        # If no chunks created (e.g., script with no functions/classes), chunk by lines
        if not chunks:
            chunks = self._chunk_generic_code(content, file_path)
            
        return chunks
    
    def _chunk_rdf(self, content: str, file_path: str, doc_type: DocumentType) -> List[CodeChunk]:
        """Chunk RDF by subject or named graphs"""
        chunks = []
        lines = content.split('\n')
        
        # Extract prefixes/namespaces
        prefixes = []
        for line in lines:
            if line.strip().startswith('@prefix') or line.strip().startswith('PREFIX'):
                prefixes.append(line)
        
        context = '\n'.join(prefixes)
        
        if doc_type == DocumentType.RDF_TURTLE:
            # Chunk by subject (everything until next subject or blank line group)
            current_chunk = []
            current_subject = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Skip prefix declarations
                if stripped.startswith('@prefix') or stripped.startswith('@base'):
                    continue
                
                # New subject detected
                if stripped and not stripped.startswith((' ', '\t', ';', ',')):
                    if current_chunk and current_subject:
                        chunks.append(CodeChunk(
                            file_path=file_path,
                            chunk_index=len(chunks),
                            content='\n'.join(current_chunk),
                            chunk_type="rdf_subject",
                            start_line=i - len(current_chunk),
                            end_line=i,
                            metadata={
                                'subject': current_subject,
                                'triple_count': len([l for l in current_chunk if l.strip().endswith('.')])
                            },
                            context=context
                        ))
                    current_chunk = [line]
                    current_subject = stripped.split()[0] if stripped else None
                else:
                    current_chunk.append(line)
            
            # Don't forget last chunk
            if current_chunk and current_subject:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content='\n'.join(current_chunk),
                    chunk_type="rdf_subject",
                    start_line=len(lines) - len(current_chunk),
                    end_line=len(lines),
                    metadata={'subject': current_subject},
                    context=context
                ))
                
        elif doc_type == DocumentType.RDF_NTRIPLES:
            # Chunk by fixed number of triples
            chunk_size = 50  # triples per chunk
            current_chunk = []
            
            for i, line in enumerate(lines):
                if line.strip():
                    current_chunk.append(line)
                    
                if len(current_chunk) >= chunk_size:
                    chunks.append(CodeChunk(
                        file_path=file_path,
                        chunk_index=len(chunks),
                        content='\n'.join(current_chunk),
                        chunk_type="rdf_triples",
                        start_line=i - len(current_chunk) + 1,
                        end_line=i,
                        metadata={'triple_count': len(current_chunk)},
                        context=""
                    ))
                    current_chunk = []
            
            # Last chunk
            if current_chunk:
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content='\n'.join(current_chunk),
                    chunk_type="rdf_triples",
                    start_line=len(lines) - len(current_chunk),
                    end_line=len(lines),
                    metadata={'triple_count': len(current_chunk)},
                    context=""
                ))
                
        return chunks
    
    def _chunk_generic_code(self, content: str, file_path: str) -> List[CodeChunk]:
        """Chunk code by functions/blocks using regex patterns"""
        chunks = []
        lines = content.split('\n')
        
        # Patterns for common code structures
        function_patterns = [
            r'^\s*(def|function|func|fn)\s+\w+',  # Python, JS, Go, Rust
            r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(',  # Java, C++
            r'^\s*\w+\s+\w+\s*\([^)]*\)\s*{',  # C-style functions
        ]
        
        class_patterns = [
            r'^\s*class\s+\w+',
            r'^\s*struct\s+\w+',
            r'^\s*interface\s+\w+',
        ]
        
        # Find function/class boundaries
        boundaries = []
        for i, line in enumerate(lines):
            for pattern in function_patterns + class_patterns:
                if re.match(pattern, line):
                    boundaries.append(i)
                    break
        
        # Create chunks based on boundaries
        if boundaries:
            for i, start in enumerate(boundaries):
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
                
                # Skip if chunk too small
                if end - start < 3:
                    continue
                    
                chunk_content = '\n'.join(lines[start:end])
                
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content=chunk_content,
                    chunk_type="code_block",
                    start_line=start,
                    end_line=end,
                    metadata={
                        'file_name': os.path.basename(file_path),
                        'lines': end - start
                    },
                    context=""
                ))
        else:
            # Fall back to fixed-size chunks
            chunk_size = 50  # lines
            for i in range(0, len(lines), chunk_size - 10):  # 10 lines overlap
                chunk_content = '\n'.join(lines[i:i + chunk_size])
                chunks.append(CodeChunk(
                    file_path=file_path,
                    chunk_index=len(chunks),
                    content=chunk_content,
                    chunk_type="code_section",
                    start_line=i,
                    end_line=min(i + chunk_size, len(lines)),
                    metadata={'file_name': os.path.basename(file_path)},
                    context=""
                ))
                
        return chunks
    
    def _chunk_text_fallback(self, content: str, file_path: str) -> List[CodeChunk]:
        """Fallback for non-code files"""
        # Similar to original text chunking but returns CodeChunk objects
        chunks = []
        lines = content.split('\n')
        chunk_size = 50  # lines
        
        for i in range(0, len(lines), chunk_size - 5):
            chunk_content = '\n'.join(lines[i:i + chunk_size])
            chunks.append(CodeChunk(
                file_path=file_path,
                chunk_index=len(chunks),
                content=chunk_content,
                chunk_type="text",
                start_line=i,
                end_line=min(i + chunk_size, len(lines)),
                metadata={'file_name': os.path.basename(file_path)},
                context=""
            ))
            
        return chunks
    
    def _get_source_segment(self, lines: List[str], node) -> str:
        """Extract source for an AST node"""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return '\n'.join(lines[node.lineno - 1:node.end_lineno])
        return ""
    
    def _partition_chunks(self, chunks: List[CodeChunk]) -> List[List[CodeChunk]]:
        """Partition chunks across workers"""
        num_partitions = min(self.max_workers, len(chunks))
        partitions = [[] for _ in range(num_partitions)]
        
        for i, chunk in enumerate(chunks):
            partition_idx = i % num_partitions
            partitions[partition_idx].append(chunk)
            
        return partitions
    
    def _score_chunks_parallel(self, 
                               query: str, 
                               partitions: List[List[CodeChunk]]) -> List[Dict]:
        """Score chunks in parallel"""
        all_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_partition = {
                executor.submit(self._score_partition, query, partition): i
                for i, partition in enumerate(partitions)
            }
            
            for future in as_completed(future_to_partition):
                partition_idx = future_to_partition[future]
                try:
                    results = future.result(timeout=30)
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error scoring partition {partition_idx}: {e}")
                    
        return all_results
    
    def _score_partition(self, query: str, partition: List[CodeChunk]) -> List[Dict]:
        """Score a partition of chunks"""
        results = []
        
        # Prepare content for scoring - include context if configured
        if self.config.include_context:
            chunk_texts = [
                f"{chunk.context}\n{chunk.content}" if chunk.context else chunk.content
                for chunk in partition
            ]
        else:
            chunk_texts = [chunk.content for chunk in partition]
        
        try:
            scores = self.model_manager.score_chunks(query, chunk_texts)
            
            for chunk, score in zip(partition, scores):
                results.append({
                    'chunk': chunk,
                    'score': score,
                    'file': chunk.file_path,
                    'type': chunk.chunk_type,
                    'location': f"lines {chunk.start_line}-{chunk.end_line}",
                    'metadata': chunk.metadata,
                    'content_preview': _make_preview(chunk.content, self.preview_chars)
                })
                
        except Exception as e:
            print(f"Error scoring partition: {e}")
            
        return results

# Test with actual code files
def test_code_retriever():
    """Test the code-aware retriever"""
    import sys
    sys.path.append('src/agent_rag/core')
    from local_model_integration import LocalModelManager
    
    config = {
        'local_model_provider': 'ollama',
        'server_url': 'http://localhost:11434',
        'local_retrieval_model': 'phi3'
    }
    manager = LocalModelManager(config)
    
    chunk_config = ChunkConfig(
        max_chunk_size=1024,
        preserve_structure=True,
        include_context=True
    )
    
    retriever = CodeAwareRetriever(
        model_manager=manager,
        chunk_config=chunk_config,
        max_workers=2
    )
    
    # Test with actual project files
    test_files = [
        'src/agent_rag/core/local_model_integration.py',
        'test_launcher.sh',  # Will be treated as generic code
        # Add any RDF files you have
    ]
    
    # Filter to existing files
    test_files = [f for f in test_files if os.path.exists(f)]
    
    queries = [
        "How does the OllamaProvider score chunks?",
        "What is the LocalModelManager class?",
        "How to handle HTTP requests?",
        "Authentication implementation"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = retriever.retrieve(
            query=query,
            file_paths=test_files,
            top_k=3,
            score_threshold=0.2
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"File: {result['metadata'].get('file_name', 'unknown')}")
            print(f"Type: {result['type']}")
            print(f"Location: {result['location']}")
            print(f"Score: {result['score']:.3f}")
            if result['type'] in ['function', 'method', 'class']:
                print(f"Name: {result['metadata'].get('name', 'unknown')}")
            print(f"Preview: {result['content_preview']}")

if __name__ == "__main__":
    test_code_retriever()
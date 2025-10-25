"""
Code Storage and Retrieval Module for Agent-Based RAG
Handles code partitioning, storage, and deduplication
"""

import os
import re
import ast
import hashlib
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import tiktoken
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    """Represents a code chunk with metadata"""
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'method', 'block', 'lines'
    language: str
    relevance: float = 0.0
    context_before: str = ""
    context_after: str = ""
    metadata: Dict[str, Any] = None

class CodeStorageDB:
    """Manages the code storage database"""
    
    def __init__(self, db_path: str = "code_storage.db"):
        self.db_path = db_path
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables
        self.conn.executescript("""
            -- Content-addressed storage for code snippets
            CREATE TABLE IF NOT EXISTS code_snippets (
                content_hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                language TEXT,
                snippet_type TEXT,
                normalized_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_language ON code_snippets(language);
            CREATE INDEX IF NOT EXISTS idx_type ON code_snippets(snippet_type);
            
            -- Metadata for code locations
            CREATE TABLE IF NOT EXISTS code_metadata (
                metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                start_column INTEGER,
                end_column INTEGER,
                version TEXT,
                context_before TEXT,
                context_after TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES code_snippets(content_hash)
            );
            
            CREATE INDEX IF NOT EXISTS idx_file_path ON code_metadata(file_path);
            CREATE INDEX IF NOT EXISTS idx_content_hash ON code_metadata(content_hash);
            
            -- Temporary retrieval tracking
            CREATE TABLE IF NOT EXISTS retrieval_sessions (
                session_id TEXT PRIMARY KEY,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS retrieved_items (
                retrieval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                metadata_id INTEGER NOT NULL,
                relevance_score REAL,
                retrieval_method TEXT,
                retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES retrieval_sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (metadata_id) REFERENCES code_metadata(metadata_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_session ON retrieved_items(session_id);
            CREATE INDEX IF NOT EXISTS idx_relevance ON retrieved_items(relevance_score DESC);
            
            -- Commentary and suggestions linked to code
            CREATE TABLE IF NOT EXISTS code_annotations (
                annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT NOT NULL,
                annotation_type TEXT,
                annotation_text TEXT,
                source_model TEXT,
                confidence REAL,
                relative_position TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES code_snippets(content_hash)
            );
            
            CREATE INDEX IF NOT EXISTS idx_annotation_hash ON code_annotations(content_hash);
            
            -- Optional: Embeddings
            CREATE TABLE IF NOT EXISTS code_embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB,
                embedding_model TEXT,
                dimension INTEGER,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (content_hash) REFERENCES code_snippets(content_hash)
            );
        """)
        
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

class CodePartitioner:
    """Handles intelligent code partitioning"""
    
    def __init__(self, max_tokens: int = 4000, overlap_ratio: float = 0.1):
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'matlab',
            '.jl': 'julia',
            '.sh': 'bash',
            '.sql': 'sql',
            '.md': 'markdown',
            '.txt': 'text'
        }
        
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, 'unknown')
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def partition_file(self, file_content: str, file_path: str) -> List[CodeChunk]:
        """Main partitioning method"""
        language = self.detect_language(file_path)
        chunks = []
        
        # Try different strategies based on language
        if language == 'python':
            chunks = self.ast_partition_python(file_content, file_path)
        elif language in ['javascript', 'typescript', 'java', 'cpp', 'c', 'csharp', 'go', 'rust']:
            chunks = self.heuristic_partition(file_content, file_path, language)
        
        # Fallback to sliding window if needed
        if not chunks:
            chunks = self.sliding_window_partition(file_content, file_path, language)
        
        # Ensure chunks aren't too large
        final_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk.content) > self.max_tokens:
                # Split large chunks
                sub_chunks = self.split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def ast_partition_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """Use AST to partition Python code"""
        chunks = []
        lines = content.split('\n')
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Get the source code for this node
                    start_line = node.lineno - 1
                    end_line = node.end_lineno or start_line
                    
                    # Include decorators if present
                    if node.decorator_list:
                        start_line = min(d.lineno - 1 for d in node.decorator_list)
                    
                    chunk_lines = lines[start_line:end_line]
                    chunk_content = '\n'.join(chunk_lines)
                    
                    # Get context
                    context_before = '\n'.join(lines[max(0, start_line-3):start_line])
                    context_after = '\n'.join(lines[end_line:min(len(lines), end_line+3)])
                    
                    chunk_type = 'class' if isinstance(node, ast.ClassDef) else 'function'
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line + 1,
                        end_line=end_line,
                        chunk_type=chunk_type,
                        language='python',
                        context_before=context_before,
                        context_after=context_after,
                        metadata={'name': node.name}
                    ))
        except SyntaxError:
            # If AST parsing fails, return empty to trigger fallback
            logger.warning(f"AST parsing failed for {file_path}")
            return []
        
        return chunks
    
    def heuristic_partition(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Heuristic-based partitioning for various languages"""
        chunks = []
        lines = content.split('\n')
        
        # Language-specific patterns
        patterns = {
            'javascript': r'^(function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\(.*?\)\s*=>)|class\s+\w+)',
            'typescript': r'^(function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?(?:function|\(.*?\)\s*=>)|class\s+\w+|interface\s+\w+)',
            'java': r'^(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:class|interface|enum|(?:\w+\s+)+\w+\s*\()',
            'cpp': r'^(?:class|struct|enum|(?:(?:inline|static|virtual|const)?\s*)*\w+(?:\s+\w+)*\s*\()',
            'go': r'^(?:func|type\s+\w+\s+(?:struct|interface))',
            'rust': r'^(?:pub\s+)?(?:fn|struct|enum|impl|trait)',
            'csharp': r'^(?:public|private|protected|internal)?\s*(?:static)?\s*(?:class|interface|struct|enum|(?:\w+\s+)+\w+\s*\()'
        }
        
        pattern = patterns.get(language)
        if not pattern:
            return []
        
        current_chunk_start = 0
        current_chunk_lines = []
        
        for i, line in enumerate(lines):
            if re.match(pattern, line.strip()) and current_chunk_lines:
                # Found a new definition, save current chunk
                chunk_content = '\n'.join(current_chunk_lines)
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_chunk_start + 1,
                    end_line=i,
                    chunk_type='block',
                    language=language,
                    context_before='',
                    context_after=''
                ))
                
                current_chunk_start = i
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
        
        # Add last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=current_chunk_start + 1,
                end_line=len(lines),
                chunk_type='block',
                language=language,
                context_before='',
                context_after=''
            ))
        
        return chunks
    
    def sliding_window_partition(self, content: str, file_path: str, language: str) -> List[CodeChunk]:
        """Fallback sliding window partitioning"""
        chunks = []
        lines = content.split('\n')
        
        # Calculate lines per chunk based on token limit
        sample_size = min(100, len(lines))
        sample_tokens = self.count_tokens('\n'.join(lines[:sample_size]))
        avg_tokens_per_line = sample_tokens / sample_size if sample_size > 0 else 10
        lines_per_chunk = int(self.max_tokens / avg_tokens_per_line)
        
        # Ensure minimum chunk size
        lines_per_chunk = max(10, lines_per_chunk)
        overlap_lines = int(lines_per_chunk * self.overlap_ratio)
        
        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            chunk_lines = lines[i:end]
            chunk_content = '\n'.join(chunk_lines)
            
            # Get context
            context_before = '\n'.join(lines[max(0, i-3):i])
            context_after = '\n'.join(lines[end:min(len(lines), end+3)])
            
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                chunk_type='lines',
                language=language,
                context_before=context_before,
                context_after=context_after
            ))
            
            # Move window with overlap
            i += lines_per_chunk - overlap_lines
            if i >= len(lines) - overlap_lines:
                break
        
        return chunks
    
    def split_large_chunk(self, chunk: CodeChunk) -> List[CodeChunk]:
        """Split a chunk that's too large"""
        lines = chunk.content.split('\n')
        total_lines = len(lines)
        
        # Estimate lines per sub-chunk
        total_tokens = self.count_tokens(chunk.content)
        ratio = self.max_tokens / total_tokens
        lines_per_subchunk = max(1, int(total_lines * ratio * 0.9))  # 90% to ensure under limit
        
        sub_chunks = []
        for i in range(0, total_lines, lines_per_subchunk):
            sub_content = '\n'.join(lines[i:i+lines_per_subchunk])
            
            sub_chunks.append(CodeChunk(
                content=sub_content,
                file_path=chunk.file_path,
                start_line=chunk.start_line + i,
                end_line=min(chunk.start_line + i + lines_per_subchunk, chunk.end_line),
                chunk_type=f"{chunk.chunk_type}_split",
                language=chunk.language,
                relevance=chunk.relevance,
                context_before=chunk.context_before if i == 0 else "",
                context_after=chunk.context_after if i + lines_per_subchunk >= total_lines else "",
                metadata=chunk.metadata
            ))
        
        return sub_chunks

class CodeRetrievalPipeline:
    """Main pipeline for code retrieval and storage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db = CodeStorageDB(config.get("db_path", "code_storage.db"))
        self.partitioner = CodePartitioner(
            max_tokens=config.get("max_chunk_tokens", 4000),
            overlap_ratio=config.get("chunk_overlap", 0.1)
        )
        self.dedup_threshold = config.get("dedup_threshold", 0.95)
        self.session_ttl_hours = config.get("session_ttl_hours", 24)
    
    def normalize_code(self, code: str) -> str:
        """Normalize code for deduplication"""
        # Remove comments (simple approach - can be improved)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)  # Python comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)  # C-style comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Block comments
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        
        return code
    
    def compute_content_hash(self, code: str) -> str:
        """Compute content-based hash"""
        normalized = self.normalize_code(code)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def snippet_exists(self, content_hash: str) -> bool:
        """Check if snippet already exists in database"""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT 1 FROM code_snippets WHERE content_hash = ?", (content_hash,))
        return cursor.fetchone() is not None
    
    def store_snippet(self, content_hash: str, chunk: CodeChunk):
        """Store a code snippet"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO code_snippets 
            (content_hash, content, language, snippet_type, normalized_content)
            VALUES (?, ?, ?, ?, ?)
        """, (
            content_hash,
            chunk.content,
            chunk.language,
            chunk.chunk_type,
            self.normalize_code(chunk.content)
        ))
        self.db.conn.commit()
    
    def store_metadata(self, content_hash: str, chunk: CodeChunk) -> int:
        """Store metadata for a code location"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO code_metadata
            (content_hash, file_path, start_line, end_line, 
             context_before, context_after, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            content_hash,
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.context_before,
            chunk.context_after,
            chunk.metadata.get('version', '') if chunk.metadata else ''
        ))
        self.db.conn.commit()
        return cursor.lastrowid
    
    def create_session(self, session_id: str, query: str):
        """Create a new retrieval session"""
        cursor = self.db.conn.cursor()
        expires_at = datetime.now() + timedelta(hours=self.session_ttl_hours)
        cursor.execute("""
            INSERT INTO retrieval_sessions (session_id, query, expires_at)
            VALUES (?, ?, ?)
        """, (session_id, query, expires_at))
        self.db.conn.commit()
    
    def record_retrieval(self, session_id: str, metadata_id: int, 
                        relevance: float, method: str = 'semantic'):
        """Record a retrieved item"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO retrieved_items
            (session_id, metadata_id, relevance_score, retrieval_method)
            VALUES (?, ?, ?, ?)
        """, (session_id, metadata_id, relevance, method))
        self.db.conn.commit()
    
    def store_retrieval(self, chunks: List[CodeChunk], session_id: str, query: str):
        """Store retrieved chunks with deduplication"""
        # Create session
        self.create_session(session_id, query)
        
        for chunk in chunks:
            content_hash = self.compute_content_hash(chunk.content)
            
            # Store snippet if new
            if not self.snippet_exists(content_hash):
                self.store_snippet(content_hash, chunk)
            
            # Store metadata
            metadata_id = self.store_metadata(content_hash, chunk)
            
            # Record retrieval
            self.record_retrieval(session_id, metadata_id, chunk.relevance)
    
    def load_retrieved_items(self, session_id: str) -> List[Dict]:
        """Load all retrieved items for a session"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT 
                ri.retrieval_id,
                ri.relevance_score,
                ri.retrieval_method,
                cm.content_hash,
                cm.file_path,
                cm.start_line,
                cm.end_line,
                cs.content,
                cs.language,
                cs.snippet_type
            FROM retrieved_items ri
            JOIN code_metadata cm ON ri.metadata_id = cm.metadata_id
            JOIN code_snippets cs ON cm.content_hash = cs.content_hash
            WHERE ri.session_id = ?
            ORDER BY ri.relevance_score DESC
        """, (session_id,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                'retrieval_id': row[0],
                'relevance': row[1],
                'method': row[2],
                'content_hash': row[3],
                'file_path': row[4],
                'start_line': row[5],
                'end_line': row[6],
                'content': row[7],
                'language': row[8],
                'snippet_type': row[9]
            })
        
        return items
    
    def prepare_for_synthesis(self, session_id: str, 
                            max_tokens: Optional[int] = None) -> str:
        """Prepare retrieved content for synthesis"""
        items = self.load_retrieved_items(session_id)
        
        if not items:
            return ""
        
        # Group by content hash
        by_hash = defaultdict(list)
        for item in items:
            by_hash[item['content_hash']].append(item)
        
        # Calculate total tokens
        if max_tokens is None:
            max_tokens = self.config.get("synthesizer_context", 10000)
        
        output_parts = []
        current_tokens = 0
        
        for content_hash, duplicates in by_hash.items():
            # Use highest relevance duplicate
            best = max(duplicates, key=lambda x: x['relevance'])
            
            # Format with metadata
            formatted = self._format_chunk_for_synthesis(best, len(duplicates))
            chunk_tokens = len(self.partitioner.tokenizer.encode(formatted))
            
            if current_tokens + chunk_tokens <= max_tokens:
                output_parts.append(formatted)
                current_tokens += chunk_tokens
            else:
                # Reached token limit
                output_parts.append(f"\n[... {len(by_hash) - len(output_parts)} more chunks omitted due to context limit ...]")
                break
        
        return "\n\n".join(output_parts)
    
    def _format_chunk_for_synthesis(self, item: Dict, duplicate_count: int) -> str:
        """Format a chunk for synthesis"""
        header = f"=== {item['file_path']} (lines {item['start_line']}-{item['end_line']}) ==="
        
        if duplicate_count > 1:
            header += f" [appears {duplicate_count} times]"
        
        return f"{header}\n{item['content']}"
    
    def store_annotation(self, content_hash: str, annotation_type: str,
                        text: str, model: str, confidence: float = 0.5,
                        position: Optional[Dict] = None):
        """Store an annotation for a code snippet"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            INSERT INTO code_annotations
            (content_hash, annotation_type, annotation_text, 
             source_model, confidence, relative_position)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            content_hash,
            annotation_type,
            text,
            model,
            confidence,
            json.dumps(position) if position else None
        ))
        self.db.conn.commit()
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            DELETE FROM retrieval_sessions
            WHERE expires_at < datetime('now')
        """)
        self.db.conn.commit()
    
    def get_storage_size(self) -> int:
        """Get database size in bytes"""
        return os.path.getsize(self.db.db_path)
    
    def close(self):
        """Close database connection"""
        self.db.close()
"""
Document Chunking Utility
Implements hierarchical chunking with overlap for the RAG system
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

import re
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import tiktoken
import sqlite3
from datetime import datetime
import logging

# --- configuration knobs (add near other constants) ---
DEFAULT_PREVIEW_CHARS = int(os.getenv("PREVIEW_CHARS", "200"))
DEFAULT_MACRO_TOKENS = int(os.getenv("MACRO_CHUNK_TOKENS", "1500"))
DEFAULT_MACRO_OVERLAP = int(os.getenv("MACRO_OVERLAP_TOKENS", "500"))
DEFAULT_MICRO_TOKENS = int(os.getenv("MICRO_CHUNK_TOKENS", "300"))
DEFAULT_MICRO_OVERLAP = int(os.getenv("MICRO_OVERLAP_TOKENS", "100"))


# --- helper for preview in exports ---
def make_preview(text: str, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
    return text[:limit] + ("..." if len(text) > limit else "")

def chunk_to_dict(c: Chunk, preview_chars: int = DEFAULT_PREVIEW_CHARS) -> Dict[str, Any]:
    d = asdict(c)
    d["preview"] = make_preview(c.text, preview_chars)
    return d

# Get database path from config
try:
    from agent_rag.config import get_config
    config = get_config()
    DEFAULT_DB_PATH = config["db_path"]
except ImportError:
    DEFAULT_DB_PATH = os.getenv("DB_PATH", "rag_index.db")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HierarchyConfig:
    macro_tokens: int = DEFAULT_MACRO_TOKENS
    macro_overlap: int = DEFAULT_MACRO_OVERLAP
    micro_tokens: int = DEFAULT_MICRO_TOKENS
    micro_overlap: int = DEFAULT_MICRO_OVERLAP
    preview_chars: int = DEFAULT_PREVIEW_CHARS

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: str
    text: str
    chunk_type: str  # 'macro' or 'micro'
    parent_id: Optional[str]  # For micro chunks
    source_file: str
    start_pos: int
    end_pos: int
    token_count: int
    metadata: Dict[str, Any]

class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, source_file: str) -> List[Chunk]:
        """Override in subclasses"""
        raise NotImplementedError

class MacroChunker(ChunkingStrategy):
    """Creates large chunks for broad topic retrieval"""
    
    def __init__(self, chunk_size: int = 1500, overlap: int = 500):
        super().__init__(chunk_size, overlap)
    
    def chunk_text(self, text: str, source_file: str) -> List[Chunk]:
        """
        Split text into macro chunks with overlap
        
        Args:
            text: The text to chunk
            source_file: Source file path
        
        Returns:
            List of macro chunks
        """
        chunks = []
        
        # Try to split on natural boundaries (paragraphs, sections)
        sections = self._split_on_boundaries(text)
        
        current_chunk = []
        current_tokens = 0
        start_pos = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            
            if current_tokens + section_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk_id = self._generate_chunk_id(chunk_text, "macro")
                    
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        chunk_type="macro",
                        parent_id=None,
                        source_file=source_file,
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk_text),
                        token_count=current_tokens,
                        metadata={
                            "section_count": len(current_chunk)
                        }
                    ))
                    
                    # Start new chunk with overlap
                    overlap_sections = self._get_overlap_sections(current_chunk)
                    current_chunk = overlap_sections + [section]
                    current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                    start_pos += len(chunk_text) - sum(len(s) for s in overlap_sections)
            else:
                current_chunk.append(section)
                current_tokens += section_tokens
        
        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text, "macro")
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunk_type="macro",
                parent_id=None,
                source_file=source_file,
                start_pos=start_pos,
                end_pos=len(text),
                token_count=current_tokens,
                metadata={
                    "section_count": len(current_chunk)
                }
            ))
        
        return chunks
    
    def _split_on_boundaries(self, text: str) -> List[str]:
        """Split text on natural boundaries like paragraphs"""
        # Split on double newlines first
        sections = text.split('\n\n')
        
        # Further split very long sections
        result = []
        for section in sections:
            if self.count_tokens(section) > self.chunk_size:
                # Split on single newlines
                subsections = section.split('\n')
                result.extend(subsections)
            else:
                result.append(section)
        
        return [s.strip() for s in result if s.strip()]
    
    def _get_overlap_sections(self, sections: List[str]) -> List[str]:
        """Get sections for overlap from the end of current chunk"""
        overlap_tokens = 0
        overlap_sections = []
        
        for section in reversed(sections):
            section_tokens = self.count_tokens(section)
            if overlap_tokens + section_tokens <= self.overlap:
                overlap_sections.insert(0, section)
                overlap_tokens += section_tokens
            else:
                break
        
        return overlap_sections
    
    def _generate_chunk_id(self, text: str, chunk_type: str) -> str:
        """Generate unique ID for chunk"""
        hash_input = f"{chunk_type}:{text[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

class MicroChunker(ChunkingStrategy):
    """Creates small chunks for fine-grained retrieval"""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 100):
        super().__init__(chunk_size, overlap)
    
    def chunk_text(self, text: str, source_file: str, 
                   parent_id: Optional[str] = None) -> List[Chunk]:
        """
        Split text into micro chunks with overlap
        
        Args:
            text: The text to chunk
            source_file: Source file path
            parent_id: ID of parent macro chunk
        
        Returns:
            List of micro chunks
        """
        chunks = []
        
        # Split on sentence boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_id = self._generate_chunk_id(chunk_text, "micro")
                    
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        chunk_type="micro",
                        parent_id=parent_id,
                        source_file=source_file,
                        start_pos=start_pos,
                        end_pos=start_pos + len(chunk_text),
                        token_count=current_tokens,
                        metadata={
                            "sentence_count": len(current_chunk)
                        }
                    ))
                    
                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap_sentences(current_chunk)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                    start_pos += len(chunk_text) - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Save final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text, "micro")
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                chunk_type="micro",
                parent_id=parent_id,
                source_file=source_file,
                start_pos=start_pos,
                end_pos=len(text),
                token_count=current_tokens,
                metadata={
                    "sentence_count": len(current_chunk)
                }
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy or NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        overlap_tokens = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _generate_chunk_id(self, text: str, chunk_type: str) -> str:
        """Generate unique ID for chunk"""
        hash_input = f"{chunk_type}:{text[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

class HierarchicalChunker:
    """
    Backward-compatible hierarchical chunker with optional persistence.
    - db_path: same default and table schema as existing implementation
    - persist: if True, init_db() is called and save_* helpers can be used
    """
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        hierarchy: Optional[HierarchyConfig] = None,
        persist: bool = True,
    ):
        self.cfg = hierarchy or HierarchyConfig()
        self.macro_chunker = MacroChunker(chunk_size=self.cfg.macro_tokens, overlap=self.cfg.macro_overlap)
        self.micro_chunker = MicroChunker(chunk_size=self.cfg.micro_tokens, overlap=self.cfg.micro_overlap)
        self.db_path = db_path
        self.persist = persist
        if self.persist:
            self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id TEXT PRIMARY KEY,
          parent_id TEXT,
          text TEXT NOT NULL,
          chunk_type TEXT NOT NULL,
          source_file TEXT NOT NULL,
          start_pos INTEGER,
          end_pos INTEGER,
          token_count INTEGER,
          metadata TEXT,
          created_at TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_parent_id ON chunks(parent_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_source_file ON chunks(source_file)")
        conn.commit()
        conn.close()

    def chunk_document(self, file_path: Path) -> Tuple[List[Chunk], List[Chunk]]:
        text = file_path.read_text(encoding="utf-8")
        source_file = str(file_path)
        macro_chunks = self.macro_chunker.chunk_text(text, source_file)
        all_micro_chunks: List[Chunk] = []
        for m in macro_chunks:
            all_micro_chunks.extend(
                self.micro_chunker.chunk_text(m.text, source_file, parent_id=m.chunk_id)
            )
        logger.info(f"Created {len(macro_chunks)} macro chunks and {len(all_micro_chunks)} micro chunks")
        return macro_chunks, all_micro_chunks

    def save_chunks(self, chunks: List[Chunk]):
        if not self.persist or not chunks:
            return
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for chunk in chunks:
            c.execute("""
            INSERT OR REPLACE INTO chunks
            (chunk_id, parent_id, text, chunk_type, source_file,
             start_pos, end_pos, token_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                chunk.parent_id,
                chunk.text,
                chunk.chunk_type,
                chunk.source_file,
                chunk.start_pos,
                chunk.end_pos,
                chunk.token_count,
                json.dumps(chunk.metadata),
                datetime.now().isoformat(),
            ))
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(chunks)} chunks to {self.db_path}")

    def get_micro_chunks_for_macro(self, macro_id: str) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
        SELECT chunk_id, text, metadata FROM chunks
        WHERE parent_id = ? AND chunk_type = 'micro'
        """, (macro_id,))
        rows = c.fetchall()
        conn.close()
        out: List[Dict[str, Any]] = []
        for cid, text, meta in rows:
            out.append({
                "chunk_id": cid,
                "text": text,
                "metadata": json.loads(meta) if meta else {},
                "preview": make_preview(text, self.cfg.preview_chars),
            })
        return out

# Utility functions for processing multiple documents
def chunk_directory(directory: Path, db_path: str = DEFAULT_DB_PATH, hierarchy: Optional[HierarchyConfig] = None):
    """
    Chunk all documents in a directory
    
    Args:
        directory: Path to directory
        db_path: Database path
    """
    chunker = HierarchicalChunker(db_path=db_path, hierarchy=hierarchy, persist=True)
    
    # Process all markdown and text files
    patterns = ['*.md', '*.txt', '*.py', '*.pl', '*.sh']
    files: List[Path] = []
    for pat in patterns:
        files.extend(directory.glob(f"**/{pat}"))
    logger.info(f"Found {len(files)} files to process")
    for fp in files:
        try:
            macro, micro = chunker.chunk_document(fp)
            chunker.save_chunks(macro)
            chunker.save_chunks(micro)
        except Exception as e:
            logger.error(f"Failed to process {fp}: {e}")

def export_chunks_to_json(db_path: str = DEFAULT_DB_PATH, output_path: str = "chunks_export.json",
                          preview_chars: int = DEFAULT_PREVIEW_CHARS):
    """Export chunks from database to JSON for inspection"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT chunk_id, parent_id, text, chunk_type, source_file, token_count, metadata FROM chunks")
    rows = c.fetchall()
    conn.close()
    out = []
    for cid, pid, text, ctype, src, tok, meta in rows:
        out.append({
            "chunk_id": cid,
            "parent_id": pid,
            "text": make_preview(text, preview_chars),
            "chunk_type": ctype,
            "source_file": src,
            "token_count": tok,
            "metadata": json.loads(meta) if meta else {},
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported {len(out)} chunks to {output_path}")

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk documents for RAG")
    parser.add_argument("path", help="File or directory to chunk")
    parser.add_argument("--db", default="rag_index.db", help="Database path")
    parser.add_argument("--export", action="store_true", 
                       help="Export chunks to JSON after processing")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        chunker = HierarchicalChunker(args.db)
        macro_chunks, micro_chunks = chunker.chunk_document(path)
        chunker.save_chunks(macro_chunks)
        chunker.save_chunks(micro_chunks)
    elif path.is_dir():
        chunk_directory(path, args.db)
    else:
        print(f"Error: {path} not found")
    
    if args.export:
        export_chunks_to_json(args.db)
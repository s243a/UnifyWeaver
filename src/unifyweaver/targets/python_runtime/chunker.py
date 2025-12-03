import hashlib
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

@dataclass
class Chunk:
    chunk_id: str
    text: str
    chunk_type: str
    parent_id: Optional[str]
    source_file: str
    token_count: int
    metadata: Dict[str, Any]

class ChunkingStrategy:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        if HAS_TIKTOKEN:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        if HAS_TIKTOKEN:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Approx for English

class MacroChunker(ChunkingStrategy):
    def chunk_text(self, text: str, source_file: str) -> List[Chunk]:
        chunks = []
        sections = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        
        for section in sections:
            section_tokens = self.count_tokens(section)
            if current_tokens + section_tokens > self.chunk_size:
                if current_chunk:
                    self._emit_chunk(chunks, current_chunk, "macro", None, source_file)
                    # Overlap logic simplified
                    current_chunk = [section]
                    current_tokens = section_tokens
            else:
                current_chunk.append(section)
                current_tokens += section_tokens
        
        if current_chunk:
            self._emit_chunk(chunks, current_chunk, "macro", None, source_file)
        return chunks

    def _emit_chunk(self, chunks, section_list, ctype, pid, src):
        text = '\n\n'.join(section_list)
        cid = hashlib.md5(text.encode()).hexdigest()[:12]
        chunks.append(Chunk(cid, text, ctype, pid, src, self.count_tokens(text), {}))

class MicroChunker(ChunkingStrategy):
    def chunk_text(self, text: str, source_file: str, parent_id: str) -> List[Chunk]:
        chunks = []
        # Split by sentence-like boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_tokens = 0
        
        for sent in sentences:
            toks = self.count_tokens(sent)
            if current_tokens + toks > self.chunk_size:
                if current_chunk:
                    self._emit_chunk(chunks, current_chunk, "micro", parent_id, source_file)
                    current_chunk = [sent]
                    current_tokens = toks
            else:
                current_chunk.append(sent)
                current_tokens += toks
                
        if current_chunk:
            self._emit_chunk(chunks, current_chunk, "micro", parent_id, source_file)
        return chunks

    def _emit_chunk(self, chunks, sent_list, ctype, pid, src):
        text = ' '.join(sent_list)
        cid = hashlib.md5(text.encode()).hexdigest()[:12]
        chunks.append(Chunk(cid, text, ctype, pid, src, self.count_tokens(text), {}))

class HierarchicalChunker:
    def __init__(self, macro_size=1000, macro_overlap=200, micro_size=250, micro_overlap=50):
        self.defaults = {
            'macro_size': macro_size,
            'macro_overlap': macro_overlap,
            'micro_size': micro_size,
            'micro_overlap': micro_overlap
        }

    def chunk(self, text: str, source_file: str, **kwargs) -> List[Chunk]:
        cfg = {**self.defaults, **kwargs}
        
        macro = MacroChunker(cfg.get('macro_size'), cfg.get('macro_overlap'))
        micro = MicroChunker(cfg.get('micro_size'), cfg.get('micro_overlap'))
        
        macros = macro.chunk_text(text, source_file)
        all_chunks = list(macros)
        for m in macros:
            micros = micro.chunk_text(m.text, source_file, m.chunk_id)
            all_chunks.extend(micros)
        return all_chunks

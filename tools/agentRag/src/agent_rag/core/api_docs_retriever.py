"""
API Documentation Retrieval Component

Implements secondary retrieval process that searches project API documentation
for relevant pydocs based on code retrieval output.

Author: Agent-Based RAG System
Date: 2025-01-31
"""

import os
import re
import ast
import sqlite3
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DocSection:
    """Represents a section of API documentation"""
    section_id: str
    file_path: str
    section_name: str
    content: str
    section_type: str  # 'class', 'function', 'module', 'overview', 'example'
    parent_section: Optional[str] = None
    keywords: Set[str] = None
    references: Set[str] = None
    relevance_score: float = 0.0


@dataclass
class ApiReference:
    """Represents a reference found in documentation"""
    symbol: str
    reference_type: str  # 'import', 'inherit', 'call', 'parameter', 'return'
    context: str


class ApiDocsIndexer:
    """Builds and maintains an index of API documentation"""
    
    def __init__(self, docs_path: str, cache_db: str = "api_docs_index.db"):
        self.docs_path = Path(docs_path)
        self.cache_db = cache_db
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for documentation index"""
        conn = sqlite3.connect(self.cache_db)
        
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_docs_sections (
            section_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            section_name TEXT NOT NULL,
            content TEXT NOT NULL,
            section_type TEXT NOT NULL,
            parent_section TEXT,
            keywords TEXT,  -- JSON array of keywords
            content_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS api_docs_references (
            reference_id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_id TEXT NOT NULL,
            referenced_symbol TEXT NOT NULL,
            reference_type TEXT NOT NULL,
            context TEXT,
            FOREIGN KEY (section_id) REFERENCES api_docs_sections(section_id)
        );
        
        CREATE TABLE IF NOT EXISTS api_docs_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_section_type ON api_docs_sections(section_type);
        CREATE INDEX IF NOT EXISTS idx_keywords ON api_docs_sections(keywords);
        CREATE INDEX IF NOT EXISTS idx_referenced_symbol ON api_docs_references(referenced_symbol);
        CREATE INDEX IF NOT EXISTS idx_reference_type ON api_docs_references(reference_type);
        """)
        
        conn.commit()
        conn.close()
    
    def index_documentation(self, force_rebuild: bool = False) -> Dict[str, int]:
        """Index all documentation files in the docs_path"""
        if not force_rebuild and self._is_index_current():
            logger.info("Documentation index is current, skipping rebuild")
            return self._get_index_stats()
        
        logger.info(f"Indexing documentation from {self.docs_path}")
        
        stats = {"sections": 0, "references": 0, "files": 0}
        
        # Clear existing data if rebuilding
        if force_rebuild:
            self._clear_index()
        
        # Index different types of documentation
        for pattern in ["*.md", "*.rst", "*.py"]:
            for doc_file in self.docs_path.rglob(pattern):
                if self._should_skip_file(doc_file):
                    continue
                
                try:
                    sections = self._extract_sections(doc_file)
                    self._store_sections(sections)
                    stats["sections"] += len(sections)
                    stats["files"] += 1
                    
                    # Extract references from sections
                    for section in sections:
                        refs = self._extract_references(section)
                        self._store_references(section.section_id, refs)
                        stats["references"] += len(refs)
                        
                except Exception as e:
                    logger.error(f"Error indexing {doc_file}: {e}")
        
        self._update_metadata("last_index_time", datetime.now().isoformat())
        self._update_metadata("docs_path", str(self.docs_path))
        
        logger.info(f"Indexing complete: {stats}")
        return stats
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if file should be skipped during indexing"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            "build",
            "dist"
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in skip_patterns)
    
    def _extract_sections(self, file_path: Path) -> List[DocSection]:
        """Extract documentation sections from a file"""
        sections = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Could not decode {file_path}, skipping")
            return sections
        
        if file_path.suffix == '.py':
            sections.extend(self._extract_python_docstrings(file_path, content))
        elif file_path.suffix in ['.md', '.rst']:
            sections.extend(self._extract_markdown_sections(file_path, content))
        
        return sections
    
    def _extract_python_docstrings(self, file_path: Path, content: str) -> List[DocSection]:
        """Extract docstrings from Python files"""
        sections = []
        
        try:
            tree = ast.parse(content)
            
            # Module-level docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                section_id = f"{file_path.stem}_module"
                sections.append(DocSection(
                    section_id=section_id,
                    file_path=str(file_path),
                    section_name=f"Module: {file_path.stem}",
                    content=module_doc,
                    section_type="module",
                    keywords=self._extract_keywords(module_doc)
                ))
            
            # Class and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node)
                    if class_doc:
                        section_id = f"{file_path.stem}_{node.name}_class"
                        sections.append(DocSection(
                            section_id=section_id,
                            file_path=str(file_path),
                            section_name=f"Class: {node.name}",
                            content=class_doc,
                            section_type="class",
                            keywords=self._extract_keywords(class_doc, [node.name])
                        ))
                
                elif isinstance(node, ast.FunctionDef):
                    func_doc = ast.get_docstring(node)
                    if func_doc:
                        section_id = f"{file_path.stem}_{node.name}_function"
                        parent = None
                        
                        # Check if this is a method inside a class
                        for parent_node in ast.walk(tree):
                            if isinstance(parent_node, ast.ClassDef) and node in ast.walk(parent_node):
                                parent = f"{file_path.stem}_{parent_node.name}_class"
                                break
                        
                        sections.append(DocSection(
                            section_id=section_id,
                            file_path=str(file_path),
                            section_name=f"Function: {node.name}",
                            content=func_doc,
                            section_type="function",
                            parent_section=parent,
                            keywords=self._extract_keywords(func_doc, [node.name])
                        ))
        
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
        
        return sections
    
    def _extract_markdown_sections(self, file_path: Path, content: str) -> List[DocSection]:
        """Extract sections from Markdown/RST files"""
        sections = []
        
        # Split by headers
        header_pattern = r'^#{1,6}\s+(.+)$'
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        current_level = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section:
                    content_text = '\n'.join(current_content).strip()
                    if content_text:
                        sections.append(DocSection(
                            section_id=current_section['id'],
                            file_path=str(file_path),
                            section_name=current_section['name'],
                            content=content_text,
                            section_type="overview",
                            parent_section=current_section.get('parent'),
                            keywords=self._extract_keywords(content_text, [current_section['name']])
                        ))
                
                # Start new section
                header_text = header_match.group(1).strip()
                level = len(line) - len(line.lstrip('#'))
                
                section_id = self._generate_section_id(file_path, header_text)
                
                # Determine parent (simplified - could be more sophisticated)
                parent = None
                if level > 1 and sections:
                    parent = sections[-1].section_id
                
                current_section = {
                    'id': section_id,
                    'name': header_text,
                    'parent': parent
                }
                current_content = []
                current_level = level
            
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            content_text = '\n'.join(current_content).strip()
            if content_text:
                sections.append(DocSection(
                    section_id=current_section['id'],
                    file_path=str(file_path),
                    section_name=current_section['name'],
                    content=content_text,
                    section_type="overview",
                    parent_section=current_section.get('parent'),
                    keywords=self._extract_keywords(content_text, [current_section['name']])
                ))
        
        return sections
    
    def _extract_keywords(self, content: str, additional_keywords: List[str] = None) -> Set[str]:
        """Extract keywords from content for search indexing"""
        keywords = set()
        
        # Add any provided additional keywords
        if additional_keywords:
            keywords.update(word.lower() for word in additional_keywords)
        
        # Extract from content using various patterns
        
        # Code-like patterns (camelCase, snake_case, etc.)
        code_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*[a-zA-Z0-9]\b'
        code_words = re.findall(code_pattern, content)
        keywords.update(word.lower() for word in code_words if len(word) > 2)
        
        # API/technical terms
        tech_pattern = r'\b(?:API|HTTP|JSON|REST|SQL|DB|URL|URI|UUID|JWT|OAuth|CORS|CRUD)\b'
        tech_words = re.findall(tech_pattern, content, re.IGNORECASE)
        keywords.update(word.lower() for word in tech_words)
        
        # Common programming words
        prog_words = ['function', 'method', 'class', 'module', 'parameter', 'return', 'exception', 
                     'error', 'response', 'request', 'data', 'config', 'settings', 'auth']
        for word in prog_words:
            if word.lower() in content.lower():
                keywords.add(word.lower())
        
        # Remove very common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        keywords = keywords - stop_words
        
        # Only keep reasonable length keywords
        keywords = {kw for kw in keywords if 2 <= len(kw) <= 50}
        
        return keywords
    
    def _extract_references(self, section: DocSection) -> List[ApiReference]:
        """Extract API references from documentation content"""
        references = []
        content = section.content
        
        # Python import patterns
        import_pattern = r'(?:from\s+(\w+(?:\.\w+)*)\s+)?import\s+(\w+(?:\s*,\s*\w+)*)'
        for match in re.finditer(import_pattern, content):
            module = match.group(1) or ""
            symbols = match.group(2).split(',')
            for symbol in symbols:
                symbol = symbol.strip()
                if symbol:
                    references.append(ApiReference(
                        symbol=f"{module}.{symbol}" if module else symbol,
                        reference_type="import",
                        context=match.group(0)
                    ))
        
        # Function call patterns
        call_pattern = r'\b(\w+)\s*\('
        for match in re.finditer(call_pattern, content):
            symbol = match.group(1)
            if symbol and not symbol[0].isupper():  # Likely function, not class
                references.append(ApiReference(
                    symbol=symbol,
                    reference_type="call",
                    context=match.group(0)
                ))
        
        # Class instantiation patterns
        class_pattern = r'\b([A-Z]\w+)\s*\('
        for match in re.finditer(class_pattern, content):
            symbol = match.group(1)
            references.append(ApiReference(
                symbol=symbol,
                reference_type="instantiate",
                context=match.group(0)
            ))
        
        return references
    
    def _generate_section_id(self, file_path: Path, section_name: str) -> str:
        """Generate unique section ID"""
        base = f"{file_path.stem}_{section_name}"
        # Clean and hash for uniqueness
        clean_base = re.sub(r'[^\w\-_]', '_', base)
        hash_suffix = hashlib.md5(f"{file_path}_{section_name}".encode()).hexdigest()[:8]
        return f"{clean_base}_{hash_suffix}"
    
    def _store_sections(self, sections: List[DocSection]):
        """Store sections in the database"""
        conn = sqlite3.connect(self.cache_db)
        
        for section in sections:
            content_hash = hashlib.md5(section.content.encode()).hexdigest()
            keywords_json = json.dumps(list(section.keywords) if section.keywords else [])
            
            conn.execute("""
            INSERT OR REPLACE INTO api_docs_sections 
            (section_id, file_path, section_name, content, section_type, 
             parent_section, keywords, content_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                section.section_id, section.file_path, section.section_name,
                section.content, section.section_type, section.parent_section,
                keywords_json, content_hash, datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    def _store_references(self, section_id: str, references: List[ApiReference]):
        """Store references in the database"""
        conn = sqlite3.connect(self.cache_db)
        
        # Clear existing references for this section
        conn.execute("DELETE FROM api_docs_references WHERE section_id = ?", (section_id,))
        
        # Insert new references
        for ref in references:
            conn.execute("""
            INSERT INTO api_docs_references 
            (section_id, referenced_symbol, reference_type, context)
            VALUES (?, ?, ?, ?)
            """, (section_id, ref.symbol, ref.reference_type, ref.context))
        
        conn.commit()
        conn.close()
    
    def _is_index_current(self) -> bool:
        """Check if index is current based on file modification times"""
        # Simplified check - could be more sophisticated
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT value FROM api_docs_metadata WHERE key = 'last_index_time'")
            result = cursor.fetchone()
            if not result:
                return False
                
            last_index = datetime.fromisoformat(result[0])
            
            # Check if any files have been modified since last index
            for doc_file in self.docs_path.rglob("*"):
                if doc_file.is_file() and not self._should_skip_file(doc_file):
                    if datetime.fromtimestamp(doc_file.stat().st_mtime) > last_index:
                        return False
            
            return True
            
        except Exception:
            return False
        finally:
            conn.close()
    
    def _clear_index(self):
        """Clear all indexed data"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute("DELETE FROM api_docs_references")
        conn.execute("DELETE FROM api_docs_sections")
        conn.commit()
        conn.close()
    
    def _update_metadata(self, key: str, value: str):
        """Update metadata in database"""
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
        INSERT OR REPLACE INTO api_docs_metadata (key, value, updated_at)
        VALUES (?, ?, ?)
        """, (key, value, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def _get_index_stats(self) -> Dict[str, int]:
        """Get statistics about the current index"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute("SELECT COUNT(*) FROM api_docs_sections")
        stats["sections"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM api_docs_references")
        stats["references"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT file_path) FROM api_docs_sections")
        stats["files"] = cursor.fetchone()[0]
        
        conn.close()
        return stats


class ApiDocsRetriever:
    """Retrieves relevant API documentation based on code retrieval results"""
    
    def __init__(self, docs_path: str, cache_db: str = "api_docs_index.db"):
        self.indexer = ApiDocsIndexer(docs_path, cache_db)
        self.cache_db = cache_db
        
        # Ensure documentation is indexed
        self.indexer.index_documentation()
    
    def retrieve_relevant_docs(self, 
                             retrieval_output: Dict,
                             max_docs: int = 5,
                             source: str = "database") -> List[DocSection]:
        """
        Retrieve relevant API documentation based on code retrieval output.
        
        Args:
            retrieval_output: Output from code retrieval (either from DB or direct)
            max_docs: Maximum number of documentation sections to return
            source: "database" or "direct" - where retrieval_output comes from
            
        Returns:
            List of relevant DocSection objects with relevance scores
        """
        if source == "database":
            return self._retrieve_from_db_output(retrieval_output, max_docs)
        elif source == "direct":
            return self._retrieve_from_direct_output(retrieval_output, max_docs)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _retrieve_from_db_output(self, db_output: Dict, max_docs: int) -> List[DocSection]:
        """Retrieve docs based on database retrieval output"""
        # Extract symbols and keywords from retrieved code
        symbols = set()
        keywords = set()
        
        # Analyze retrieved_items if present
        if "retrieved_items" in db_output:
            for item in db_output["retrieved_items"]:
                # Extract from content
                content = item.get("content", "")
                symbols.update(self._extract_symbols_from_code(content))
                keywords.update(self._extract_keywords_from_code(content))
        
        return self._search_docs(symbols, keywords, max_docs)
    
    def _retrieve_from_direct_output(self, direct_output: Dict, max_docs: int) -> List[DocSection]:
        """Retrieve docs based on direct retrieval output"""
        symbols = set()
        keywords = set()
        
        # Handle different direct output formats
        if "context" in direct_output:
            contexts = direct_output["context"]
            if isinstance(contexts, list):
                for context in contexts:
                    symbols.update(self._extract_symbols_from_code(str(context)))
                    keywords.update(self._extract_keywords_from_code(str(context)))
        
        return self._search_docs(symbols, keywords, max_docs)
    
    def _extract_symbols_from_code(self, code: str) -> Set[str]:
        """Extract API symbols (functions, classes, etc.) from code"""
        symbols = set()
        
        try:
            # Try to parse as Python
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    symbols.add(node.id)
                elif isinstance(node, ast.Attribute):
                    symbols.add(node.attr)
                elif isinstance(node, ast.FunctionDef):
                    symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    symbols.add(node.name)
        except:
            # Fall back to regex patterns
            # Function/method calls
            func_pattern = r'\b(\w+)\s*\('
            symbols.update(re.findall(func_pattern, code))
            
            # Class names (capitalized)
            class_pattern = r'\b([A-Z]\w+)\b'
            symbols.update(re.findall(class_pattern, code))
            
            # Attribute access
            attr_pattern = r'\.(\w+)'
            symbols.update(re.findall(attr_pattern, code))
        
        return symbols
    
    def _extract_keywords_from_code(self, code: str) -> Set[str]:
        """Extract keywords from code for documentation search"""
        keywords = set()
        
        # Common programming keywords that might appear in docs
        prog_keywords = ['auth', 'config', 'database', 'api', 'endpoint', 'middleware',
                        'validation', 'error', 'response', 'request', 'handler', 'service']
        
        for keyword in prog_keywords:
            if keyword in code.lower():
                keywords.add(keyword)
        
        # Extract camelCase and snake_case identifiers
        identifier_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, code)
        keywords.update(id.lower() for id in identifiers if len(id) > 3)
        
        return keywords
    
    def _search_docs(self, symbols: Set[str], keywords: Set[str], max_docs: int) -> List[DocSection]:
        """Search documentation index for relevant sections"""
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.cursor()
        
        doc_scores = {}  # section_id -> score
        
        # Search by direct symbol references
        if symbols:
            symbol_list = list(symbols)
            placeholders = ','.join('?' * len(symbol_list))
            cursor.execute(f"""
            SELECT DISTINCT s.section_id, s.file_path, s.section_name, s.content, 
                   s.section_type, s.parent_section, s.keywords
            FROM api_docs_sections s
            JOIN api_docs_references r ON s.section_id = r.section_id
            WHERE r.referenced_symbol IN ({placeholders})
            """, symbol_list)
            
            for row in cursor.fetchall():
                section_id = row[0]
                doc_scores[section_id] = doc_scores.get(section_id, 0) + 2.0  # High score for symbol matches
        
        # Search by keywords in content and keyword index
        if keywords:
            for keyword in keywords:
                # Search in content
                cursor.execute("""
                SELECT section_id, file_path, section_name, content, 
                       section_type, parent_section, keywords
                FROM api_docs_sections
                WHERE content LIKE ? OR section_name LIKE ?
                """, (f'%{keyword}%', f'%{keyword}%'))
                
                for row in cursor.fetchall():
                    section_id = row[0]
                    # Score based on where the keyword appears
                    if keyword.lower() in row[2].lower():  # section_name
                        doc_scores[section_id] = doc_scores.get(section_id, 0) + 1.5
                    else:  # content
                        doc_scores[section_id] = doc_scores.get(section_id, 0) + 1.0
        
        # Get all relevant sections with their scores
        relevant_sections = []
        all_section_ids = list(doc_scores.keys())
        
        if all_section_ids:
            placeholders = ','.join('?' * len(all_section_ids))
            cursor.execute(f"""
            SELECT section_id, file_path, section_name, content, 
                   section_type, parent_section, keywords
            FROM api_docs_sections
            WHERE section_id IN ({placeholders})
            """, all_section_ids)
            
            for row in cursor.fetchall():
                section_id = row[0]
                keywords_list = json.loads(row[6]) if row[6] else []
                
                section = DocSection(
                    section_id=section_id,
                    file_path=row[1],
                    section_name=row[2],
                    content=row[3],
                    section_type=row[4],
                    parent_section=row[5],
                    keywords=set(keywords_list),
                    relevance_score=doc_scores[section_id]
                )
                relevant_sections.append(section)
        
        conn.close()
        
        # Sort by relevance score and return top results
        relevant_sections.sort(key=lambda x: x.relevance_score, reverse=True)
        return relevant_sections[:max_docs]
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the documentation index"""
        return self.indexer._get_index_stats()
    
    def rebuild_index(self) -> Dict[str, int]:
        """Force rebuild of the documentation index"""
        return self.indexer.index_documentation(force_rebuild=True)


# Integration functions for the orchestrator

def integrate_api_docs_retrieval(retrieval_output: Dict, 
                                config: Dict,
                                source: str = "database") -> Dict:
    """
    Integrate API documentation retrieval into the main pipeline.
    
    This function can be called from extended_orchestrator.py after code retrieval.
    """
    if not config.get("enable_api_docs_retrieval", False):
        return retrieval_output
    
    docs_path = config.get("api_docs_path", "./docs/api")
    cache_db = config.get("api_docs_db", "api_docs_index.db")
    max_docs = config.get("api_docs_max_results", 5)
    
    if not os.path.exists(docs_path):
        logger.warning(f"API docs path {docs_path} does not exist, skipping API docs retrieval")
        return retrieval_output
    
    try:
        retriever = ApiDocsRetriever(docs_path, cache_db)
        relevant_docs = retriever.retrieve_relevant_docs(retrieval_output, max_docs, source)
        
        # Add API docs to the retrieval output
        if relevant_docs:
            api_docs_context = []
            for doc in relevant_docs:
                api_docs_context.append({
                    "section_name": doc.section_name,
                    "content": doc.content,
                    "file_path": doc.file_path,
                    "section_type": doc.section_type,
                    "relevance_score": doc.relevance_score
                })
            
            retrieval_output["api_documentation"] = {
                "shard": "api_docs",
                "context": api_docs_context,
                "confidence": 0.8,  # API docs are generally reliable
                "metadata": {
                    "retrieval_method": "api_docs_secondary",
                    "total_docs": len(relevant_docs)
                }
            }
        
    except Exception as e:
        logger.error(f"Error in API docs retrieval: {e}")
    
    return retrieval_output


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="API Documentation Retrieval Tool")
    parser.add_argument("--docs-path", default="./docs/api", help="Path to API documentation")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--stats", action="store_true", help="Show index statistics")
    
    args = parser.parse_args()
    
    retriever = ApiDocsRetriever(args.docs_path)
    
    if args.rebuild:
        print("Rebuilding documentation index...")
        stats = retriever.rebuild_index()
        print(f"Index rebuilt: {stats}")
    
    if args.stats:
        stats = retriever.get_index_stats()
        print(f"Index statistics: {stats}")
"""
Obsidian Integration for Agent-Based RAG
Processes prompts from Obsidian markdown files and manages artifacts
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
import uuid
import sqlite3
import asyncio
from datetime import datetime
from slugify import slugify
from typing import Optional, Dict, Any, Tuple
import logging

# Get configuration
try:
    from agent_rag.config import get_config
    config = get_config()
    DEFAULT_DB_PATH = config["db_path"]
    DEFAULT_ORCHESTRATOR_URL = f"http://localhost:8001/query"
except ImportError:
    DEFAULT_DB_PATH = os.getenv("DB_PATH", "rag_index.db")
    DEFAULT_ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8001") + "/query"


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObsidianProcessor:
    """Handles Obsidian vault operations for the RAG system"""
    
    def __init__(self, vault_path: Path, db_path: str = "rag_index.db"):
        self.vault_path = Path(vault_path)
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for tracking answers"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            CREATE TABLE IF NOT EXISTS obsidian_answers (
                answer_id TEXT PRIMARY KEY,
                note_path TEXT NOT NULL,
                artifact_path TEXT,
                timestamp TEXT,
                model TEXT,
                query TEXT,
                metadata TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_last_unanswered_prompt(self, note_text: str) -> Optional[str]:
        """
        Find the last [!PROMPT] block that doesn't have an [!ANSWER META] after it
        
        Args:
            note_text: Content of the Obsidian note
        
        Returns:
            The prompt text if found, None otherwise
        """
        # Find all [!PROMPT] blocks
        prompt_pattern = r'>\[!PROMPT\]\n>(.*?)(?=\n(?!>)|$)'
        prompts = list(re.finditer(prompt_pattern, note_text, re.DOTALL | re.MULTILINE))
        
        if not prompts:
            return None
        
        # Check each prompt from last to first
        for prompt in reversed(prompts):
            # Get text after this prompt
            after_text = note_text[prompt.end():]
            
            # Check if there's an [!ANSWER META] block after it
            if not re.search(r'>\[!ANSWER META\]', after_text[:500]):  # Check next 500 chars
                # Clean and return the prompt text
                prompt_text = prompt.group(1)
                # Remove the '>' prefix from continuation lines
                prompt_text = re.sub(r'\n>', '\n', prompt_text).strip()
                return prompt_text
        
        return None
    
    def make_artifact_name(self, base_name: Optional[str], prompt: str, 
                          artifact_folder: Path) -> str:
        """
        Generate a versioned artifact filename
        
        Args:
            base_name: Suggested name from Claude/LLM
            prompt: The prompt text (for generating name if needed)
            artifact_folder: Folder where artifact will be saved
        
        Returns:
            Versioned filename like 'auth_review_v1.md'
        """
        # Determine base stem and extension
        if base_name:
            stem = Path(base_name).stem
            ext = Path(base_name).suffix or '.md'
        elif prompt:
            # Create slug from first few words of prompt
            words = prompt.split()[:4]
            stem = slugify(' '.join(words))
            ext = '.md'
        else:
            # Fallback to timestamp
            stem = f"artifact_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ext = '.md'
        
        # Find next available version number
        existing_versions = []
        for file in artifact_folder.glob(f"{stem}_v*{ext}"):
            match = re.search(r'_v(\d+)', file.stem)
            if match:
                existing_versions.append(int(match.group(1)))
        
        next_version = max(existing_versions, default=0) + 1
        return f"{stem}_v{next_version}{ext}"
    
    def get_artifact_folder(self, note_path: Path) -> Path:
        """
        Get or create the artifact folder for a chat note
        
        Args:
            note_path: Path to the chat note
        
        Returns:
            Path to the artifact folder
        """
        # Extract number prefix from chat filename
        chat_filename = note_path.stem  # e.g., "01 info_econ"
        prefix_match = re.match(r'^(\d+)', chat_filename)
        
        if prefix_match:
            prefix = prefix_match.group(1)
            artifact_folder_name = f"{prefix} artifacts"
        else:
            # No number prefix, use full name
            artifact_folder_name = f"{chat_filename} artifacts"
        
        # Create artifacts folder in same directory as chat note
        chat_folder = note_path.parent
        artifact_folder = chat_folder / artifact_folder_name
        
        # Create if doesn't exist
        artifact_folder.mkdir(exist_ok=True, parents=True)
        
        return artifact_folder
    
    def append_answer(self, note_path: Path, answer_text: str, 
                     artifact_link: str, metadata: Dict[str, Any]) -> None:
        """
        Append answer with metadata to the note
        
        Args:
            note_path: Path to the Obsidian note
            answer_text: The answer content
            artifact_link: Relative link to the artifact file
            metadata: Additional metadata (model, timestamp, etc.)
        """
        # Read current note content
        note_text = note_path.read_text(encoding='utf-8')
        
        # Build metadata block
        meta_lines = []
        meta_lines.append(f"link: [[{artifact_link}]]")
        meta_lines.append(f"model: {metadata.get('model', 'unknown')}")
        meta_lines.append(f"timestamp: {metadata.get('timestamp', datetime.now().isoformat())}")
        meta_lines.append(f"answer_id: {metadata.get('answer_id', str(uuid.uuid4()))}")
        
        # Add any additional metadata
        for key, value in metadata.items():
            if key not in ['model', 'timestamp', 'answer_id']:
                meta_lines.append(f"{key}: {value}")
        
        meta_block = '\n> '.join(meta_lines)
        
        # Append to note
        updated_note = note_text + f"\n\n>[!ANSWER META]\n> {meta_block}\n\n{answer_text}\n"
        
        # Write back
        note_path.write_text(updated_note, encoding='utf-8')
    
    def save_artifact(self, content: str, artifact_folder: Path, 
                     suggested_name: Optional[str], prompt: str) -> str:
        """
        Save artifact content to a file
        
        Args:
            content: The artifact content
            artifact_folder: Folder to save in
            suggested_name: Suggested filename from LLM
            prompt: Original prompt (for name generation)
        
        Returns:
            Relative path to the artifact from the chat folder
        """
        # Generate versioned filename
        artifact_name = self.make_artifact_name(suggested_name, prompt, artifact_folder)
        
        # Save artifact
        artifact_path = artifact_folder / artifact_name
        artifact_path.write_text(content, encoding='utf-8')
        
        # Return relative path from chat folder
        return f"{artifact_folder.name}/{artifact_name}"
    
    def log_to_db(self, answer_id: str, note_path: str, artifact_path: str,
                  model: str, query: str, metadata: Dict = None):
        """Log answer to database for future retrieval"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO obsidian_answers 
            (answer_id, note_path, artifact_path, timestamp, model, query, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            answer_id,
            str(note_path),
            artifact_path,
            datetime.now().isoformat(),
            model,
            query,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()

class PromptProcessor:
    """Processes prompts through the orchestration pipeline"""
    
    def __init__(self, orchestrator_url: str = "http://localhost:8001/query"):
        self.orchestrator_url = orchestrator_url
    
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Send prompt to orchestrator and get response
        
        Args:
            prompt: The prompt text
        
        Returns:
            Dict with answer, metadata, and optional artifact content
        """
        import aiohttp
        
        # Call orchestrator API
        async with aiohttp.ClientSession() as session:
            payload = {
                "query": prompt,
                "shards": []  # Will be populated by orchestrator from context
            }
            
            async with session.post(self.orchestrator_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Extract artifact if code is present
                    artifact_content = None
                    artifact_name = None
                    
                    answer_text = result.get("answer", "")
                    
                    # Check if answer contains code blocks (simple heuristic)
                    if "```" in answer_text:
                        # Extract code blocks as artifact
                        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', 
                                                answer_text, re.DOTALL)
                        if code_blocks:
                            artifact_content = "\n\n".join(code_blocks)
                            # Try to extract a suggested name from the answer
                            name_match = re.search(r'(?:file|script|class|function):\s*(\S+)', 
                                                  answer_text, re.IGNORECASE)
                            if name_match:
                                artifact_name = name_match.group(1)
                    
                    return {
                        "answer": answer_text,
                        "artifact_content": artifact_content,
                        "artifact_name": artifact_name,
                        "metadata": result.get("metadata", {}),
                        "master_prompt": result.get("master_prompt", "")
                    }
                else:
                    # Fallback for API errors
                    return {
                        "answer": f"Error: Orchestrator returned status {response.status}",
                        "artifact_content": None,
                        "artifact_name": None,
                        "metadata": {"error": True}
                    }

async def process_note(note_path: Path, orchestrator_url: Optional[str] = None):
    """
    Main function to process a single Obsidian note
    
    Args:
        note_path: Path to the Obsidian note
        orchestrator_url: Optional orchestrator API URL
    """
    # Initialize processors
    obsidian = ObsidianProcessor(note_path.parent.parent)  # Vault is 2 levels up
    
    if orchestrator_url:
        prompt_processor = PromptProcessor(orchestrator_url)
    else:
        # Use default URL
        prompt_processor = PromptProcessor()
    
    # Read note
    note_text = note_path.read_text(encoding='utf-8')
    
    # Find last unanswered prompt
    prompt = obsidian.get_last_unanswered_prompt(note_text)
    
    if not prompt:
        logger.info("No unanswered prompts found in note")
        return
    
    logger.info(f"Processing prompt: {prompt[:100]}...")
    
    # Process through orchestrator
    result = await prompt_processor.process_prompt(prompt)
    
    # Prepare artifact if needed
    artifact_link = None
    if result["artifact_content"]:
        artifact_folder = obsidian.get_artifact_folder(note_path)
        artifact_link = obsidian.save_artifact(
            result["artifact_content"],
            artifact_folder,
            result["artifact_name"],
            prompt
        )
        logger.info(f"Saved artifact: {artifact_link}")
    
    # Append answer to note
    metadata = result["metadata"]
    metadata["has_artifact"] = bool(artifact_link)
    
    obsidian.append_answer(
        note_path,
        result["answer"],
        artifact_link or "No artifact",
        metadata
    )
    
    # Log to database
    answer_id = metadata.get("answer_id", str(uuid.uuid4()))
    obsidian.log_to_db(
        answer_id,
        str(note_path),
        artifact_link,
        metadata.get("model", "unknown"),
        prompt,
        metadata
    )
    
    logger.info(f"Answer appended to note with ID: {answer_id}")

# CLI interface
async def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process Obsidian notes with Agent RAG")
    parser.add_argument("note_path", help="Path to the Obsidian note")
    parser.add_argument("--orchestrator-url", 
                       default="http://localhost:8001/query",
                       help="Orchestrator API URL")
    parser.add_argument("--watch", action="store_true",
                       help="Watch for changes and process automatically")
    
    args = parser.parse_args()
    
    note_path = Path(args.note_path)
    if not note_path.exists():
        print(f"Error: Note not found: {note_path}")
        return
    
    if args.watch:
        # Watch mode - process whenever file changes
        import time
        last_modified = 0
        
        print(f"Watching {note_path} for changes...")
        while True:
            current_modified = note_path.stat().st_mtime
            if current_modified > last_modified:
                last_modified = current_modified
                print(f"Change detected at {datetime.now()}")
                await process_note(note_path, args.orchestrator_url)
            time.sleep(2)  # Check every 2 seconds
    else:
        # Process once
        await process_note(note_path, args.orchestrator_url)

if __name__ == "__main__":
    asyncio.run(main())
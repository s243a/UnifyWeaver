"""
Gemini Flash Retrieval Service
Flask service that uses Gemini Flash for shard-based retrieval
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
import re
from flask import Flask, request, jsonify
import google.generativeai as genai
from typing import List, Dict, Any
import logging

# Load configuration
try:
    from agent_rag.config import get_config
    config = get_config()
    genai.configure(api_key=config["roles"]["retriever"]["api_key"])
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    genai.configure(api_key=os.environ.get("RETRIEVER_API_KEY") or os.environ["GEMINI_API_KEY"])


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Configure Gemini API
#genai.configure(api_key=os.environ["GEMINI_API_KEY"])
#TODO: Let MODEL be configurable.
MODEL = "gemini-2.0-flash-exp"  # Latest Gemini Flash model

class GeminiRetriever:
    """Handles retrieval using Gemini Flash"""
    
    def __init__(self, model_name: str = MODEL):
        self.model = genai.GenerativeModel(model_name)
    
    def retrieve_from_shard(self, query: str, shard_name: str, 
                           shard_docs: List[str], top_k: int = 3) -> Dict:
        """
        Retrieve relevant chunks from a shard using Gemini Flash
        
        Args:
            query: The search query
            shard_name: Name of the shard being searched
            shard_docs: List of document chunks in this shard
            top_k: Number of top results to return
        
        Returns:
            Dict with shard name, relevant contexts, and confidence
        """
        
        # Build the retrieval prompt
        prompt = f"""You are a retrieval agent for the '{shard_name}' shard of a codebase/knowledge base.

TASK:
Given the query and the provided context chunks, select the top {top_k} most relevant chunks.
Consider both direct keyword matches and semantic relevance.

QUERY: {query}

CONTEXT CHUNKS:
"""
        
        # Add numbered chunks for easier reference
        for i, chunk in enumerate(shard_docs):
            prompt += f"\n[CHUNK {i}]: {chunk[:500]}..."  # Truncate long chunks
        
        prompt += f"""

OUTPUT REQUIREMENTS:
Return a JSON object with this exact format:
{{
    "shard": "{shard_name}",
    "selected_indices": [list of chunk indices],
    "context": [list of selected chunk texts],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why these chunks were selected"
}}

Ensure the JSON is valid and can be parsed."""
        
        try:
            # Call Gemini Flash
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            text = response.text.strip()
            
            # Clean up the response (remove markdown code fences if present)
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            
            # Parse JSON
            result = json.loads(text)
            
            # Validate and populate context if indices are provided
            if "selected_indices" in result and not result.get("context"):
                result["context"] = [
                    shard_docs[idx] for idx in result["selected_indices"]
                    if idx < len(shard_docs)
                ]
            
            # Ensure required fields
            result.setdefault("shard", shard_name)
            result.setdefault("confidence", 0.5)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {text}")
            
            # Fallback: return first few chunks
            return {
                "shard": shard_name,
                "context": shard_docs[:top_k] if shard_docs else [],
                "confidence": 0.3,
                "reasoning": "Fallback: JSON parsing failed, returning first chunks"
            }
            
        except Exception as e:
            logger.error(f"Gemini retrieval failed: {e}")
            return {
                "shard": shard_name,
                "context": [],
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }

class MultiShardRetriever:
    """Handles retrieval across multiple shards in parallel"""
    
    def __init__(self, model_name: str = MODEL):
        self.retriever = GeminiRetriever(model_name)
    
    def retrieve(self, query: str, shards: List[Dict]) -> Dict:
        """
        Retrieve from multiple shards
        
        Args:
            query: Search query
            shards: List of dicts with 'shard_name' and 'shard_docs'
        
        Returns:
            Combined results from all shards
        """
        
        results = []
        
        for shard in shards:
            shard_result = self.retriever.retrieve_from_shard(
                query=query,
                shard_name=shard.get("shard_name", "unknown"),
                shard_docs=shard.get("shard_docs", [])
            )
            results.append(shard_result)
        
        # Combine results
        return {
            "query": query,
            "results": results,
            "total_shards": len(shards)
        }

# Initialize retriever
multi_retriever = MultiShardRetriever()

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL})

@app.route("/gemini-flash-retrieve", methods=["POST"])
def retrieve_single_shard():
    """
    Single shard retrieval endpoint
    
    Expected JSON payload:
    {
        "query": "search query",
        "shard_name": "name of shard",
        "shard_docs": ["chunk1", "chunk2", ...]
    }
    """
    
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    
    query = data.get("query")
    shard_name = data.get("shard_name", "unknown")
    shard_docs = data.get("shard_docs", [])
    
    if not query:
        return jsonify({"error": "Missing required field: query"}), 400
    
    if not shard_docs:
        return jsonify({"error": "No shard_docs provided"}), 400
    
    # Perform retrieval
    retriever = GeminiRetriever()
    result = retriever.retrieve_from_shard(query, shard_name, shard_docs)
    
    return jsonify(result)

@app.route("/gemini-multi-retrieve", methods=["POST"])
def retrieve_multiple_shards():
    """
    Multiple shard retrieval endpoint
    
    Expected JSON payload:
    {
        "query": "search query",
        "shards": [
            {"shard_name": "shard1", "shard_docs": ["chunk1", "chunk2"]},
            {"shard_name": "shard2", "shard_docs": ["chunk3", "chunk4"]}
        ]
    }
    """
    
    data = request.get_json()
    
    # Validate input
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    
    query = data.get("query")
    shards = data.get("shards", [])
    
    if not query:
        return jsonify({"error": "Missing required field: query"}), 400
    
    if not shards:
        return jsonify({"error": "No shards provided"}), 400
    
    # Perform retrieval
    result = multi_retriever.retrieve(query, shards)
    
    return jsonify(result)

@app.route("/analyze-context", methods=["POST"])
def analyze_context():
    """
    Analyze context relevance and suggest connections
    
    Expected JSON payload:
    {
        "chunks": ["chunk1", "chunk2", "chunk3"],
        "task": "Find connections between these chunks"
    }
    """
    
    data = request.get_json()
    chunks = data.get("chunks", [])
    task = data.get("task", "Analyze these chunks")
    
    if not chunks:
        return jsonify({"error": "No chunks provided"}), 400
    
    prompt = f"""Task: {task}

Chunks to analyze:
"""
    for i, chunk in enumerate(chunks):
        prompt += f"\n[CHUNK {i}]: {chunk}"
    
    prompt += """

Provide a JSON response with:
{
    "connections": ["connection1", "connection2"],
    "summary": "Brief summary of insights",
    "relevance_score": 0.0-1.0
}"""
    
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean and parse
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        result = json.loads(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Context analysis failed: {e}")
        return jsonify({
            "error": str(e),
            "connections": [],
            "summary": "Analysis failed",
            "relevance_score": 0.0
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=8000, debug=True)
    
    # For production, use:
    # gunicorn -w 4 -b 0.0.0.0:8000 gemini_retriever:app
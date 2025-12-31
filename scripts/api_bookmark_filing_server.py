#!/usr/bin/env python3
"""
REST API Server for Bookmark Filing with Fuzzy Boost.

Exposes bookmark filing as a REST API that can be consumed by any HTTP client.
Supports fuzzy logic boosting for fine-tuning candidate rankings.

Usage:
    # Start the API server
    python3 scripts/api_bookmark_filing_server.py

    # With custom port
    python3 scripts/api_bookmark_filing_server.py --port 8080

Endpoints:
    POST /api/candidates     - Get semantic search candidates
    POST /api/file           - Get LLM filing recommendation
    POST /api/dual-objective - Get dual-objective scored candidates
    GET  /api/health         - Health check

Example requests:
    # Get candidates with fuzzy boost
    curl -X POST http://localhost:5000/api/candidates \
        -H "Content-Type: application/json" \
        -d '{"bookmark_title": "bash scripting tutorial", "boost_or": "bash:0.9,shell:0.5"}'

    # File a bookmark
    curl -X POST http://localhost:5000/api/file \
        -H "Content-Type: application/json" \
        -d '{"bookmark_title": "Neural network tutorial", "provider": "claude"}'
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Check for flask
try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    print("Flask not installed. Run: pip install flask", file=sys.stderr)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
try:
    from fuzzy_boost import boost_filing_candidates
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

from bookmark_filing_assistant import (
    file_bookmark,
    get_semantic_candidates,
    get_dual_objective_candidates,
    rebuild_tree_output,
    FilingResult
)

MODEL_PATH = Path("models/pearltrees_federated_single.pkl")
DATA_PATH = Path("reports/pearltrees_targets_full_multi_account.jsonl")


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "fuzzy_boost_available": HAS_FUZZY,
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists()
        })

    @app.route('/api/candidates', methods=['POST'])
    def get_candidates():
        """
        Get semantic search candidates for a bookmark.

        Request body:
            bookmark_title: str (required)
            top_k: int (default: 10)
            boost_and: str (optional) - AND boost spec
            boost_or: str (optional) - OR boost spec
            filters: list[str] (optional) - Filter specs
            blend_alpha: float (default: 0.7)

        Returns:
            JSON with tree, candidates, and metadata
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        bookmark_title = data.get('bookmark_title', '')
        if not bookmark_title:
            return jsonify({"error": "bookmark_title is required"}), 400

        top_k = data.get('top_k', 10)
        boost_and = data.get('boost_and')
        boost_or = data.get('boost_or')
        filters = data.get('filters')
        blend_alpha = data.get('blend_alpha', 0.7)

        # Fetch more if fuzzy boost is requested
        fetch_k = top_k * 2 if (boost_and or boost_or or filters) else top_k

        try:
            tree_output, candidates = get_semantic_candidates(
                bookmark_title, MODEL_PATH, fetch_k,
                tree_mode=True, data_path=DATA_PATH
            )

            # Apply fuzzy boost if requested
            if HAS_FUZZY and candidates and (boost_and or boost_or or filters):
                candidates = boost_filing_candidates(
                    candidates,
                    boost_and=boost_and,
                    boost_or=boost_or,
                    filters=filters,
                    blend_alpha=blend_alpha,
                    top_k=top_k
                )
                tree_output = rebuild_tree_output(candidates)
            elif len(candidates) > top_k:
                candidates = candidates[:top_k]

            return jsonify({
                "tree": tree_output,
                "candidates": candidates,
                "bookmark": bookmark_title,
                "top_k": top_k,
                "fuzzy_boost": {
                    "enabled": HAS_FUZZY and bool(boost_and or boost_or or filters),
                    "boost_and": boost_and,
                    "boost_or": boost_or,
                    "filters": filters,
                    "blend_alpha": blend_alpha
                } if (boost_and or boost_or or filters) else None
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/dual-objective', methods=['POST'])
    def get_dual_objective():
        """
        Get candidates using dual-objective scoring.

        Request body:
            bookmark_title: str (required)
            top_k: int (default: 10)
            alpha: float (default: 0.7) - Blend weight

        Returns:
            JSON with tree, candidates, and metadata
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        bookmark_title = data.get('bookmark_title', '')
        if not bookmark_title:
            return jsonify({"error": "bookmark_title is required"}), 400

        top_k = data.get('top_k', 10)
        alpha = data.get('alpha', 0.7)

        try:
            tree_output, candidates = get_dual_objective_candidates(
                bookmark_title, top_k=top_k, alpha=alpha
            )

            return jsonify({
                "tree": tree_output,
                "candidates": candidates,
                "bookmark": bookmark_title,
                "alpha": alpha,
                "top_k": top_k
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/file', methods=['POST'])
    def file_bookmark_endpoint():
        """
        Get LLM recommendation for where to file a bookmark.

        Request body:
            bookmark_title: str (required)
            bookmark_url: str (optional)
            provider: str (default: "claude") - claude, gemini, openai, ollama
            top_k: int (default: 10)
            boost_and: str (optional)
            boost_or: str (optional)
            filters: list[str] (optional)
            blend_alpha: float (default: 0.7)

        Returns:
            JSON with selected folder and reasoning
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        bookmark_title = data.get('bookmark_title', '')
        if not bookmark_title:
            return jsonify({"error": "bookmark_title is required"}), 400

        bookmark_url = data.get('bookmark_url')
        provider = data.get('provider', 'claude')
        top_k = data.get('top_k', 10)
        boost_and = data.get('boost_and')
        boost_or = data.get('boost_or')
        filters = data.get('filters')
        blend_alpha = data.get('blend_alpha', 0.7)

        # Determine model based on provider
        if provider == 'claude':
            llm_model = 'sonnet'
        elif provider == 'gemini':
            llm_model = 'gemini-2.0-flash'
        elif provider == 'openai':
            llm_model = 'gpt-4o-mini'
        else:
            llm_model = 'llama3.1'

        try:
            result = file_bookmark(
                bookmark_title,
                bookmark_url,
                model_path=MODEL_PATH,
                data_path=DATA_PATH,
                provider=provider,
                llm_model=llm_model,
                top_k=top_k,
                boost_and=boost_and,
                boost_or=boost_or,
                filters=filters,
                blend_alpha=blend_alpha
            )

            if result:
                return jsonify({
                    "selected_folder": result.selected_folder,
                    "rank": result.rank,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "tree_id": result.tree_id,
                    "provider": result.provider,
                    "model": result.model,
                    "fuzzy_boost": {
                        "boost_and": boost_and,
                        "boost_or": boost_or,
                        "filters": filters,
                        "blend_alpha": blend_alpha
                    } if (boost_and or boost_or or filters) else None
                })
            else:
                return jsonify({"error": "Failed to get recommendation"}), 500

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/boost/parse', methods=['POST'])
    def parse_boost_spec():
        """
        Parse and validate a boost specification.

        Request body:
            spec: str - Boost spec to parse (e.g., "bash:0.9,shell:0.5")
            mode: str (default: "dist_or") - and, or, dist_or, union

        Returns:
            JSON with parsed terms and weights
        """
        if not HAS_FUZZY:
            return jsonify({"error": "Fuzzy boost module not available"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        spec = data.get('spec', '')
        mode = data.get('mode', 'dist_or')

        try:
            from fuzzy_boost import parse_boost_spec
            result = parse_boost_spec(spec, mode)
            return jsonify({
                "terms": result.terms,
                "mode": result.mode,
                "base_weight": result.base_weight
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


def main():
    """Run the API server."""
    if not HAS_FLASK:
        print("Flask required. Install with: pip install flask")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="REST API server for bookmark filing with fuzzy boost"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")

    args = parser.parse_args()

    app = create_app()

    print(f"Starting Bookmark Filing API Server...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Fuzzy Boost: {'enabled' if HAS_FUZZY else 'disabled'}")
    print(f"  Model: {MODEL_PATH}")
    print()
    print("Endpoints:")
    print("  POST /api/candidates     - Get semantic candidates")
    print("  POST /api/file           - Get LLM recommendation")
    print("  POST /api/dual-objective - Get dual-objective candidates")
    print("  POST /api/boost/parse    - Parse boost spec")
    print("  GET  /api/health         - Health check")
    print()

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

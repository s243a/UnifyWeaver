#!/usr/bin/env python3
"""
MCP Server for Bookmark Filing.

Exposes bookmark filing as an MCP tool that can be used by any MCP-compatible client.

Usage:
    # Start the MCP server
    python3 scripts/mcp_bookmark_filing_server.py

    # Or run with uvx (if published)
    uvx mcp-bookmark-filing

The server exposes these tools:
- get_filing_candidates: Get semantic search candidates for a bookmark
- file_bookmark: Get LLM recommendation for where to file a bookmark
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import Any

# Check for mcp package
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("MCP package not installed. Run: pip install mcp", file=sys.stderr)


# Import our inference functions
sys.path.insert(0, str(Path(__file__).parent))

MODEL_PATH = Path("models/pearltrees_federated_single.pkl")
DATA_PATH = Path("reports/pearltrees_targets_full_multi_account.jsonl")


def get_filing_candidates_sync(
    bookmark_title: str,
    top_k: int = 10,
    tree_mode: bool = True
) -> dict:
    """Get semantic candidates for filing a bookmark."""
    import subprocess
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "infer_pearltrees_federated.py"),
        "--model", str(MODEL_PATH),
        "--query", bookmark_title,
        "--top-k", str(top_k),
    ]
    
    if tree_mode:
        cmd.append("--tree")
        cmd.extend(["--data", str(DATA_PATH)])
    else:
        cmd.append("--json")
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        return {"error": result.stderr, "candidates": [], "tree": ""}
    
    output = result.stdout.strip()
    
    # Also get JSON for structured data
    cmd_json = cmd.copy()
    if "--tree" in cmd_json:
        cmd_json.remove("--tree")
        if "--data" in cmd_json:
            idx = cmd_json.index("--data")
            cmd_json.pop(idx)  # remove --data
            cmd_json.pop(idx)  # remove path
    cmd_json.append("--json")
    
    result_json = subprocess.run(cmd_json, capture_output=True, text=True, timeout=60)
    candidates = []
    if result_json.returncode == 0:
        try:
            # Find JSON in output
            for line in result_json.stdout.split('\n'):
                if line.strip().startswith('['):
                    candidates = json.loads(line.strip())
                    break
        except json.JSONDecodeError:
            pass
    
    return {
        "tree": output if tree_mode else "",
        "candidates": candidates,
        "bookmark": bookmark_title,
        "top_k": top_k
    }


def create_mcp_server():
    """Create and configure the MCP server."""
    if not HAS_MCP:
        return None
    
    server = Server("bookmark-filing")
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_filing_candidates",
                description="Get semantic search candidates for where to file a bookmark in Pearltrees. Returns a hierarchical tree of candidate folders with scores.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bookmark_title": {
                            "type": "string",
                            "description": "The title or description of the bookmark to file"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of candidates to return (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["bookmark_title"]
                }
            ),
            Tool(
                name="file_bookmark",
                description="Get LLM recommendation for where to file a bookmark. Uses semantic search to find candidates, then asks an LLM to make the final selection.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "bookmark_title": {
                            "type": "string",
                            "description": "The title or description of the bookmark"
                        },
                        "bookmark_url": {
                            "type": "string",
                            "description": "Optional URL of the bookmark"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["claude", "gemini", "openai", "ollama"],
                            "description": "LLM provider to use for final selection",
                            "default": "claude"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of candidates to consider",
                            "default": 10
                        }
                    },
                    "required": ["bookmark_title"]
                }
            )
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "get_filing_candidates":
            bookmark_title = arguments.get("bookmark_title", "")
            top_k = arguments.get("top_k", 10)
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: get_filing_candidates_sync(bookmark_title, top_k, tree_mode=True)
            )
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "file_bookmark":
            bookmark_title = arguments.get("bookmark_title", "")
            bookmark_url = arguments.get("bookmark_url")
            provider = arguments.get("provider", "claude")
            top_k = arguments.get("top_k", 10)
            
            # Import and call the filing assistant
            from bookmark_filing_assistant import file_bookmark as fb
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: fb(
                    bookmark_title,
                    bookmark_url,
                    model_path=MODEL_PATH,
                    data_path=DATA_PATH,
                    provider=provider,
                    llm_model="sonnet" if provider == "claude" else "gemini-2.0-flash",
                    top_k=top_k
                )
            )
            
            if result:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "selected_folder": result.selected_folder,
                        "rank": result.rank,
                        "score": result.score,
                        "reasoning": result.reasoning,
                        "tree_id": result.tree_id,
                        "provider": result.provider,
                        "model": result.model
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "Failed to get recommendation"})
                )]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    return server


async def main():
    """Run the MCP server."""
    if not HAS_MCP:
        print("MCP package required. Install with: pip install mcp")
        sys.exit(1)
    
    server = create_mcp_server()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())

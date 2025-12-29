#!/usr/bin/env python3
"""
Bookmark Filing Assistant - LLM Integration.

Uses semantic search to get top-k candidate folders, then calls an LLM
to make the final selection based on hierarchical context.

Supports multiple LLM backends:
- Claude CLI (coding agent - cheapest with subscription)
- Gemini CLI (coding agent)
- OpenAI API
- Anthropic API
- Local models (Ollama)

Usage:
    # Using Claude CLI (default)
    python3 scripts/bookmark_filing_assistant.py \
        --bookmark "Neural network tutorial for beginners" \
        --provider claude

    # Using Gemini CLI
    python3 scripts/bookmark_filing_assistant.py \
        --bookmark "Quantum computing paper" \
        --provider gemini

    # Using OpenAI API
    python3 scripts/bookmark_filing_assistant.py \
        --bookmark "Machine learning paper" \
        --provider openai --model gpt-4o

    # Using local Ollama
    python3 scripts/bookmark_filing_assistant.py \
        --bookmark "Deep learning course" \
        --provider ollama --model llama3.1

    # Interactive mode
    python3 scripts/bookmark_filing_assistant.py --interactive
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "unifyweaver" / "targets" / "python_runtime"))


FILING_PROMPT_TEMPLATE = '''You are a bookmark filing assistant. Your task is to select the BEST folder to file a new bookmark.

## New Bookmark
Title: {bookmark_title}
{bookmark_url}

## Candidate Folders (Top {top_k} by semantic similarity)

The following tree shows candidate folders marked with ★. The hierarchy shows the folder structure:

```
{tree_output}
```

{related_pearls}

{existing_contents}
## Instructions
1. Examine the bookmark title and URL (if provided)
2. Look at ALL starred (★) candidates and their hierarchical context
3. Consider existing bookmarks in each folder - file similar items together
4. Select the SINGLE BEST folder for this bookmark
5. Consider:
   - How specific is the folder vs how specific is the bookmark?
   - Does the bookmark fit better in a parent or child folder?
   - What existing content is already in the folder?
   - Semantic match between bookmark topic and folder name

## Response Format
Respond with ONLY the following JSON (no explanation):
{{
    "selected_folder": "<exact folder name>",
    "rank": <number>,
    "reasoning": "<brief one-sentence explanation>"
}}
'''

EXISTING_CONTENTS_TEMPLATE = '''
## Existing Bookmarks in Candidate Folders

{folder_contents}
'''


def get_existing_pearls(
    candidates: List[dict],
    db_path: Optional[Path] = None,
    max_pearls_per_folder: int = 5
) -> str:
    """
    Get existing bookmarks in candidate folders from the database.
    
    Returns formatted string showing existing contents.
    """
    if not db_path or not db_path.exists():
        return ""
    
    try:
        from importer import PtMultiAccountImporter
    except ImportError:
        return ""
    
    try:
        importer = PtMultiAccountImporter(str(db_path))
        
        folder_contents = []
        for c in candidates[:5]:  # Top 5 candidates only
            tree_id = c.get('tree_id', '')
            title = c.get('title', '')
            
            if not tree_id:
                continue
            
            pearls = importer.get_pearls_in_tree(tree_id, limit=max_pearls_per_folder)
            
            if pearls:
                pearl_titles = [p['data'].get('title', 'Untitled')[:60] for p in pearls]
                folder_contents.append(f"**{title}** (#{c.get('rank', '?')}): {', '.join(pearl_titles)}")
        
        importer.close()
        
        if folder_contents:
            return EXISTING_CONTENTS_TEMPLATE.format(
                folder_contents='\n'.join(folder_contents)
            )
    except Exception as e:
        print(f"Warning: Could not load existing pearls: {e}", file=sys.stderr)
    
    return ""


@dataclass
class FilingResult:
    """Result of filing recommendation."""
    selected_folder: str
    rank: int
    score: float
    reasoning: str
    tree_id: str
    provider: str
    model: str


def call_claude_cli(prompt: str, model: str = "sonnet", timeout: int = 60) -> Optional[str]:
    """Call Claude CLI with the given prompt."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd())
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"  Claude CLI error: {result.stderr[:200]}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  Claude CLI not found. Install with: npm install -g @anthropics/claude-code", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None


def call_gemini_cli(prompt: str, model: str = "gemini-2.0-flash", timeout: int = 120) -> Optional[str]:
    """Call Gemini CLI with the given prompt."""
    try:
        result = subprocess.run(
            ["gemini", "-p", prompt, "-m", model, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd())
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"  Gemini CLI error: {result.stderr[:200]}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  Gemini CLI not found. Install with: npm install -g @anthropics/gemini", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None


def call_openai_api(prompt: str, model: str = "gpt-4o-mini", timeout: int = 60) -> Optional[str]:
    """Call OpenAI API."""
    try:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        print("  OpenAI package not installed. Run: pip install openai", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  OpenAI API error: {e}", file=sys.stderr)
        return None


def call_anthropic_api(prompt: str, model: str = "claude-3-5-sonnet-20241022", timeout: int = 60) -> Optional[str]:
    """Call Anthropic API directly."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except ImportError:
        print("  Anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Anthropic API error: {e}", file=sys.stderr)
        return None


def call_ollama(prompt: str, model: str = "llama3.1", timeout: int = 120) -> Optional[str]:
    """Call local Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"  Ollama error: {result.stderr[:200]}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("  Ollama not found. Install from: https://ollama.ai", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None


def call_llm(prompt: str, provider: str, model: str, timeout: int = 60) -> Optional[str]:
    """Route to appropriate LLM provider."""
    if provider == "claude":
        return call_claude_cli(prompt, model, timeout)
    elif provider == "gemini":
        return call_gemini_cli(prompt, model, timeout)
    elif provider == "openai":
        return call_openai_api(prompt, model, timeout)
    elif provider == "anthropic":
        return call_anthropic_api(prompt, model, timeout)
    elif provider == "ollama":
        return call_ollama(prompt, model, timeout)
    else:
        print(f"  Unknown provider: {provider}", file=sys.stderr)
        return None


def get_semantic_candidates(
    bookmark_title: str,
    model_path: Path,
    top_k: int = 10,
    tree_mode: bool = True,
    data_path: Optional[Path] = None
) -> Tuple[str, List[dict]]:
    """Get semantic candidates using the federated model."""
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "infer_pearltrees_federated.py"),
        "--model", str(model_path),
        "--query", bookmark_title,
        "--top-k", str(top_k),
    ]
    
    if tree_mode:
        cmd.append("--tree")
        if data_path:
            cmd.extend(["--data", str(data_path)])
    else:
        cmd.append("--json")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Inference error: {result.stderr}", file=sys.stderr)
        return "", []
    
    output = result.stdout.strip()
    
    # Also get JSON version for metadata
    cmd_json = [
        sys.executable,
        str(Path(__file__).parent / "infer_pearltrees_federated.py"),
        "--model", str(model_path),
        "--query", bookmark_title,
        "--top-k", str(top_k),
        "--json"
    ]
    
    result_json = subprocess.run(cmd_json, capture_output=True, text=True)
    candidates = []
    if result_json.returncode == 0:
        try:
            # Parse JSON from output (skip log lines)
            for line in result_json.stdout.split('\n'):
                if line.strip().startswith('['):
                    candidates = json.loads(line.strip())
                    break
            if not candidates:
                # Try parsing entire output
                candidates = json.loads(result_json.stdout.strip())
        except json.JSONDecodeError:
            pass
    
    return output, candidates


def parse_llm_response(response: str, candidates: List[dict]) -> Optional[FilingResult]:
    """Parse LLM response to extract selected folder."""
    try:
        # Try to extract JSON from response
        # Handle cases where LLM adds extra text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            selected_folder = data.get("selected_folder", "")
            rank = data.get("rank", 0)
            reasoning = data.get("reasoning", "")
            
            # Find matching candidate
            tree_id = ""
            score = 0.0
            for c in candidates:
                if c.get("title") == selected_folder or selected_folder in c.get("title", ""):
                    tree_id = c.get("tree_id", "")
                    score = c.get("score", 0.0)
                    break
            
            return FilingResult(
                selected_folder=selected_folder,
                rank=rank,
                score=score,
                reasoning=reasoning,
                tree_id=tree_id,
                provider="",
                model=""
            )
    except json.JSONDecodeError:
        pass
    
    return None


def file_bookmark(
    bookmark_title: str,
    bookmark_url: Optional[str] = None,
    model_path: Path = Path("models/pearltrees_federated_single.pkl"),
    pearl_model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    provider: str = "claude",
    llm_model: str = "sonnet",
    top_k: int = 10,
    timeout: int = 60
) -> Optional[FilingResult]:
    """
    Get LLM recommendation for where to file a bookmark.
    
    Args:
        bookmark_title: Title or description of the bookmark
        bookmark_url: Optional URL
        model_path: Path to the federated model
        data_path: Path to JSONL data for tree display
        db_path: Path to SQLite DB with existing pearls
        provider: LLM provider (claude, gemini, openai, anthropic, ollama)
        llm_model: Model name for the provider
        top_k: Number of candidates to consider
        timeout: Timeout for LLM calls
        
    Returns:
        FilingResult with selected folder and reasoning
    """
    
    # Get semantic candidates
    print(f"Getting semantic candidates for: {bookmark_title[:50]}...")
    tree_output, candidates = get_semantic_candidates(
        bookmark_title, model_path, top_k, tree_mode=True, data_path=data_path
    )
    
    if not tree_output or not candidates:
        print("Failed to get candidates", file=sys.stderr)
        return None
    
    # Get existing pearls if DB available
    existing_contents = ""
    if db_path:
        existing_contents = get_existing_pearls(candidates, db_path)

    # Get related pearls if model available
    related_pearls = ""
    if pearl_model_path and pearl_model_path.exists():
        print(f"Finding similar pearls using {pearl_model_path}...")
        # Use specific query style for pearl model
        pearl_query = f"locate_url({bookmark_title})"
        
        # Use existing get_semantic_candidates but we need raw json, so tree_mode=False.
        # But get_semantic_candidates returns (str, list). tree_mode=False returns empty str for tree output.
        _, pearl_candidates = get_semantic_candidates(
            pearl_query, pearl_model_path, top_k=5, tree_mode=False
        )
        
        if pearl_candidates:
            related_pearls = "## Similar Existing Bookmarks\n"
            for pc in pearl_candidates:
                score = pc.get('score', 0)
                title = pc.get('title', 'Unknown')
                path = pc.get('path')
                # If path different from title, show it
                if path and path != title:
                    related_pearls += f"- {title} (in {path}) [Score: {score:.2f}]\n"
                else:
                    related_pearls += f"- {title} [Score: {score:.2f}]\n"
    
    # Build prompt
    url_line = f"URL: {bookmark_url}" if bookmark_url else ""
    prompt = FILING_PROMPT_TEMPLATE.format(
        bookmark_title=bookmark_title,
        bookmark_url=url_line,
        top_k=top_k,
        tree_output=tree_output,
        related_pearls=related_pearls,
        existing_contents=existing_contents
    )
    
    # Call LLM
    print(f"Asking {provider}/{llm_model} for recommendation...")
    response = call_llm(prompt, provider, llm_model, timeout)
    
    if not response:
        print("LLM call failed", file=sys.stderr)
        return None
    
    # Parse response
    result = parse_llm_response(response, candidates)
    if result:
        result.provider = provider
        result.model = llm_model
    
    return result


def interactive_mode(
    model_path: Path,
    pearl_model_path: Optional[Path],
    data_path: Optional[Path],
    db_path: Optional[Path],
    provider: str,
    llm_model: str,
    top_k: int
):
    """Run in interactive mode."""
    print("\n=== Bookmark Filing Assistant ===")
    print(f"Model: {model_path}")
    print(f"LLM: {provider}/{llm_model}")
    if db_path and db_path.exists():
        print(f"DB: {db_path} (existing pearls enabled)")
    print("\nEnter bookmark titles to get filing recommendations.")
    print("Type 'quit' to exit.\n")
    
    while True:
        try:
            title = input("Bookmark> ").strip()
            if not title:
                continue
            if title.lower() in ("quit", "exit", "q"):
                break
            
            url = input("URL (optional)> ").strip() or None
            
            result = file_bookmark(
                title, url,
                model_path=model_path,
                pearl_model_path=pearl_model_path,
                data_path=data_path,
                db_path=db_path,
                provider=provider,
                llm_model=llm_model,
                top_k=top_k
            )
            
            if result:
                print(f"\n✓ Recommended folder: {result.selected_folder}")
                print(f"  Rank: #{result.rank} | Score: {result.score:.3f}")
                print(f"  Reasoning: {result.reasoning}")
                print(f"  Tree ID: {result.tree_id}")
            else:
                print("\n✗ Could not get recommendation")
            
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Bookmark filing assistant using semantic search + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--bookmark", type=str, help="Bookmark title to file")
    parser.add_argument("--url", type=str, help="Bookmark URL (optional)")
    parser.add_argument("--model", type=Path, 
                       default=Path("models/pearltrees_federated_single.pkl"),
                       help="Path to federated model (Tree)")
    parser.add_argument("--pearl-model", type=Path, 
                       default=None,
                       help="Path to federated model with pearls (optional)")
    parser.add_argument("--data", type=Path, 
                       default=Path("reports/pearltrees_targets_full_multi_account.jsonl"),
                       help="Path to JSONL data for tree display")
    parser.add_argument("--db", type=Path, 
                       default=None,
                       help="Path to SQLite DB with existing pearls (enables showing folder contents)")
    parser.add_argument("--provider", type=str, default="claude",
                       choices=["claude", "gemini", "openai", "anthropic", "ollama"],
                       help="LLM provider")
    parser.add_argument("--llm-model", type=str, default="sonnet",
                       help="Model for the provider (e.g., sonnet, gpt-4o, llama3.1)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of semantic candidates")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Timeout for LLM calls")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--json", action="store_true",
                       help="Output result as JSON")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model, args.pearl_model, args.data, args.db, args.provider, args.llm_model, args.top_k)
    elif args.bookmark:
        result = file_bookmark(
            args.bookmark,
            args.url,
            model_path=args.model,
            pearl_model_path=args.pearl_model,
            data_path=args.data,
            db_path=args.db,
            provider=args.provider,
            llm_model=args.llm_model,
            top_k=args.top_k,
            timeout=args.timeout
        )
        
        if result:
            if args.json:
                print(json.dumps({
                    "selected_folder": result.selected_folder,
                    "rank": result.rank,
                    "score": result.score,
                    "reasoning": result.reasoning,
                    "tree_id": result.tree_id,
                    "provider": result.provider,
                    "model": result.model
                }, indent=2))
            else:
                print(f"\n✓ Recommended folder: {result.selected_folder}")
                print(f"  Rank: #{result.rank} | Score: {result.score:.3f}")
                print(f"  Reasoning: {result.reasoning}")
                print(f"  Tree ID: {result.tree_id}")
        else:
            print("Failed to get recommendation", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

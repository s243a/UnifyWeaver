#!/usr/bin/env python3
"""
Interactive Bookmark Filing Agent Launcher

A unified launcher for the bookmark filing assistant with multiple modes.

Usage:
    python3 scripts/launch_filing_agent.py              # Interactive REPL with Claude
    python3 scripts/launch_filing_agent.py --agent      # Launch Claude Code agent
    python3 scripts/launch_filing_agent.py --ollama     # Use local Ollama
    python3 scripts/launch_filing_agent.py --gemini     # Use Gemini CLI
    python3 scripts/launch_filing_agent.py "My bookmark" # Quick single file
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Default paths
DEFAULT_MODEL = PROJECT_ROOT / "models" / "pearltrees_federated_single.pkl"
DEFAULT_DATA = PROJECT_ROOT / "reports" / "pearltrees_targets_full_multi_account.jsonl"
DEFAULT_DB = PROJECT_ROOT / "reports" / "pearltrees.db"


def main():
    parser = argparse.ArgumentParser(
        description="Launch the interactive bookmark filing agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Conversational agent (default)
  %(prog)s "bash tutorial"          # Agent with initial bookmark
  %(prog)s --direct                 # Direct mode: one-shot LLM per bookmark
  %(prog)s --ollama                 # Direct mode with local Ollama
  %(prog)s --semantic-only          # Show candidates without LLM

Direct mode commands:
  boost <term:weight,...>           Set OR boost terms
  boost-and <term:weight,...>       Set AND boost terms
  filter <predicate:value>          Add filter (e.g., in_subtree:Unix)
  status                            Show current boosts/filters
  clear                             Clear all boosts/filters
  quit                              Exit

Note: Direct mode is lighter on usage but lacks session state.
      Use conversational mode (default) for complex filing decisions.
"""
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--direct", action="store_true",
                            help="Direct mode: one-shot LLM calls per bookmark (no file access)")
    mode_group.add_argument("--ollama", action="store_true",
                            help="Direct mode with local Ollama")
    mode_group.add_argument("--gemini", action="store_true",
                            help="Direct mode with Gemini CLI")
    mode_group.add_argument("--openai", action="store_true",
                            help="Direct mode with OpenAI API")
    mode_group.add_argument("--semantic-only", action="store_true",
                            help="Show semantic candidates without LLM")

    # Optional positional for quick single bookmark
    parser.add_argument("bookmark", nargs="?", help="Bookmark to file (skips interactive)")

    # Model options
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Semantic model path")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM model name (provider-specific)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of candidates (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Blend alpha for hybrid scoring (default: 0.7)")

    # Fuzzy boost options
    parser.add_argument("--boost-or", type=str, default=None,
                        help="OR boost: 'term:weight,...' (any term can match)")
    parser.add_argument("--boost-and", type=str, default=None,
                        help="AND boost: 'term:weight,...' (all must match)")
    parser.add_argument("--filter", type=str, action="append", dest="filters",
                        help="Filter: 'predicate:value' (e.g., in_subtree:Unix)")

    args = parser.parse_args()

    # Check if any direct mode flag is set
    direct_mode = args.direct or args.ollama or args.gemini or args.openai or args.semantic_only

    # Default: conversational agent mode (unless direct mode requested)
    if not direct_mode:
        run_claude_code_agent(args.bookmark)
        return

    # Direct mode: determine provider
    if args.ollama:
        provider = "ollama"
        llm_model = args.llm_model or "llama3.1"
    elif args.gemini:
        provider = "gemini"
        llm_model = args.llm_model or "gemini"
    elif args.openai:
        provider = "openai"
        llm_model = args.llm_model or "gpt-4o-mini"
    elif args.semantic_only:
        provider = None
        llm_model = None
    else:
        # --direct with Claude CLI
        provider = "claude"
        llm_model = args.llm_model or "sonnet"

    # Check model exists for direct modes
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        print("Train with: python3 scripts/train_pearltrees_federated.py", file=sys.stderr)
        sys.exit(1)

    if args.semantic_only:
        run_semantic_only(args)
    elif args.bookmark:
        run_single_file(args, provider, llm_model)
    else:
        run_interactive(args, provider, llm_model)


def run_claude_code_agent(bookmark: str = None):
    """Launch Claude Code as a conversational bookmark filing agent."""
    agent_file = PROJECT_ROOT / "docs" / "ai-skills" / "bookmark-filing-agent.md"

    prompt = """You are the Bookmark Filing Assistant. Your role is defined in: docs/ai-skills/bookmark-filing-agent.md

Please read that file first to understand your capabilities.

Key commands available:
- `python3 scripts/infer_pearltrees_federated.py --model models/pearltrees_federated_single.pkl --query "TITLE" --top-k 10 --tree` - Get candidates
- `python3 scripts/bookmark_filing_assistant.py --bookmark "TITLE" --provider claude` - Full LLM recommendation
"""

    if bookmark:
        prompt += f'\nThe user wants to file this bookmark: "{bookmark}"\n\nPlease run the semantic search and provide your recommendation.'

    os.chdir(PROJECT_ROOT)
    os.execvp("claude", ["claude", prompt])


def run_semantic_only(args):
    """Run semantic search without LLM."""
    from scripts.infer_pearltrees_federated import (
        FederatedInferenceEngine, format_candidates, build_merged_tree, format_tree
    )
    import json

    engine = FederatedInferenceEngine(args.model)

    # Load data for tree display
    data = []
    if DEFAULT_DATA.exists():
        with open(DEFAULT_DATA) as f:
            data = [json.loads(line) for line in f]

    print("\n=== Semantic Candidate Search ===")
    print(f"Model: {args.model.name}")
    print(f"Top-k: {args.top_k}")
    print("\nEnter bookmark titles to see candidates. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query> ").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                break

            candidates = engine.search(query, top_k=args.top_k)

            if data:
                tree = build_merged_tree(candidates, data)
                print(format_tree(tree))
            else:
                print(format_candidates(candidates))
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_single_file(args, provider, llm_model):
    """File a single bookmark."""
    from scripts.bookmark_filing_assistant import file_bookmark

    result = file_bookmark(
        args.bookmark,
        url=None,
        model_path=args.model,
        pearl_model_path=None,
        data_path=DEFAULT_DATA if DEFAULT_DATA.exists() else None,
        db_path=DEFAULT_DB if DEFAULT_DB.exists() else None,
        provider=provider,
        llm_model=llm_model,
        top_k=args.top_k,
        boost_and=args.boost_and,
        boost_or=args.boost_or,
        filters=args.filters,
        blend_alpha=args.alpha
    )

    if result:
        print(f"\n✓ Recommended folder: {result.selected_folder}")
        print(f"  Rank: #{result.rank} | Score: {result.score:.3f}")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Tree ID: {result.tree_id}")
    else:
        print("Failed to get recommendation", file=sys.stderr)
        sys.exit(1)


def run_interactive(args, provider, llm_model):
    """Run interactive REPL mode."""
    from scripts.bookmark_filing_assistant import interactive_mode

    db_path = DEFAULT_DB if DEFAULT_DB.exists() else None
    data_path = DEFAULT_DATA if DEFAULT_DATA.exists() else None

    interactive_mode(
        model_path=args.model,
        pearl_model_path=None,
        data_path=data_path,
        db_path=db_path,
        provider=provider,
        llm_model=llm_model,
        top_k=args.top_k,
        boost_and=args.boost_and,
        boost_or=args.boost_or,
        filters=args.filters,
        blend_alpha=args.alpha
    )


if __name__ == "__main__":
    main()

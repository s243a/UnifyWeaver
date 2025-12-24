#!/usr/bin/env python3
"""
Generate tailored answers for Q-A pairs using Claude CLI.

For each (question, base_answer) pair, generates a reworded answer that:
1. Directly addresses the specific question
2. Maintains semantic equivalence with the base answer
3. Has natural variation in phrasing

Usage:
    python scripts/generate_tailored_answers.py --input training-data/expanded
    python scripts/generate_tailored_answers.py --input training-data/expanded --batch-size 5
    python scripts/generate_tailored_answers.py --input training-data/expanded --model opus
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional


PROMPT_TEMPLATE = '''You are helping create training data for UnifyWeaver's semantic search system.

PROJECT STRUCTURE:
- Main project: ~/Projects/UnifyWeaver/context/claude/UnifyWeaver/
  - src/unifyweaver/ - Core source code (Prolog, Python runtime)
  - docs/proposals/ - Design documents and proposals
  - playbooks/ - Examples and tutorials
  - training-data/ - Q&A training data

- Education (submodule): ~/Projects/UnifyWeaver/context/claude/UnifyWeaver/education/
  - book-01-foundations/ through book-14-ai-training/
  - Each book has chapters covering specific topics

PROJECT CONTEXT:
- UnifyWeaver is a Prolog-to-target-language compiler (Bash, Python, Go, Rust, C#, PowerShell, SQL)
- The project uses semantic search with LDA projection for Q&A retrieval
- Key features: recursive query compilation, data sources, cross-target glue, security/firewall

Given a question and a base answer, rewrite the answer to:
1. Directly address the specific question (start by addressing what they asked)
2. Keep all the essential technical information from the base answer
3. Use slightly different wording/phrasing for diversity
4. Be concise but complete
5. If mentioning where to find more info, reference specific paths (e.g., "See education/book-02-bash-target/ for details")

Source File: {source_file}

Question: {question}

Base Answer:
{base_answer}

Rewrite the answer to best match this specific question. Output ONLY the rewritten answer text, no explanations or prefixes.'''


def call_claude_cli(prompt: str, model: str = "sonnet", timeout: int = 60) -> Optional[str]:
    """Call claude CLI with the given prompt."""
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
            print(f"  CLI error: {result.stderr[:200]}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout}s", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None


def call_gemini_cli(prompt: str, model: str = "gemini-3-flash-preview", timeout: int = 120) -> Optional[str]:
    """Call gemini CLI with the given prompt."""
    try:
        # Gemini CLI: -p for print mode, -m for model, --output-format text for clean output
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
    except Exception as e:
        print(f"  Exception: {e}", file=sys.stderr)
        return None


def call_llm_cli(prompt: str, provider: str = "claude", model: str = "sonnet", timeout: int = 60) -> Optional[str]:
    """Call LLM CLI based on provider."""
    if provider == "gemini":
        return call_gemini_cli(prompt, model, timeout)
    else:
        return call_claude_cli(prompt, model, timeout)


def generate_tailored_answer(
    question: str,
    base_answer: str,
    source_file: str = "",
    provider: str = "claude",
    model: str = "sonnet"
) -> Optional[str]:
    """Generate a tailored answer for a specific question."""
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        base_answer=base_answer[:2000],  # Truncate very long answers
        source_file=source_file or "N/A"
    )
    return call_llm_cli(prompt, provider, model)


def extract_text(field: Any) -> str:
    """Extract text from a field that may be string or dict with 'text' key."""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict):
        return field.get("text", str(field))
    else:
        return str(field) if field else ""


def process_batch(pairs: List[Dict[str, Any]], provider: str, model: str, delay: float = 0.5) -> List[Dict[str, Any]]:
    """Process a batch of pairs, generating tailored answers."""
    results = []
    for i, pair in enumerate(pairs):
        question = extract_text(pair.get("question", ""))
        base_answer = extract_text(pair.get("answer", ""))
        source_file = pair.get("answer_source", "") or pair.get("source_file", "")

        print(f"  [{i+1}/{len(pairs)}] {question[:50]}...")

        tailored = generate_tailored_answer(question, base_answer, source_file, provider, model)

        if tailored:
            new_pair = pair.copy()
            new_pair["original_answer"] = base_answer
            new_pair["answer"] = tailored
            new_pair["answer_tailored"] = True
            results.append(new_pair)
        else:
            # Keep original if generation fails
            pair_copy = pair.copy()
            pair_copy["answer_tailored"] = False
            results.append(pair_copy)

        # Small delay to avoid rate limiting
        if delay > 0 and i < len(pairs) - 1:
            time.sleep(delay)

    return results


def process_file(
    input_path: Path,
    output_path: Path,
    provider: str = "claude",
    model: str = "sonnet",
    batch_size: int = 10,
    delay: float = 0.5,
    skip_existing: bool = True
) -> Dict[str, int]:
    """Process a single JSONL file."""

    # Load existing results if skip_existing
    existing = {}
    if skip_existing and output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        existing[record.get("pair_id", "")] = record
                    except:
                        pass

    # Load input pairs
    pairs = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    pairs.append(json.loads(line))
                except:
                    pass

    # Filter out already processed
    to_process = []
    already_done = []
    for pair in pairs:
        pair_id = pair.get("pair_id", "")
        if pair_id in existing and existing[pair_id].get("answer_tailored"):
            already_done.append(existing[pair_id])
        else:
            to_process.append(pair)

    if not to_process:
        print(f"  All {len(already_done)} pairs already processed")
        return {"processed": 0, "skipped": len(already_done), "total": len(pairs)}

    print(f"  Processing {len(to_process)} pairs ({len(already_done)} already done)")

    # Process in batches
    results = already_done.copy()
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(to_process) + batch_size - 1)//batch_size}")
        batch_results = process_batch(batch, provider, model, delay)
        results.extend(batch_results)

        # Save progress after each batch
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

    processed = sum(1 for r in results if r.get("answer_tailored"))
    return {"processed": processed, "skipped": len(already_done), "total": len(pairs)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate tailored answers using Claude CLI"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training-data/expanded"),
        help="Input directory with expanded JSONL files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: input/tailored/)"
    )
    parser.add_argument(
        "--provider",
        default="claude",
        choices=["claude", "gemini"],
        help="LLM provider to use (claude or gemini)"
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Model to use (claude: sonnet/opus/haiku, gemini: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pairs per batch"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between CLI calls (seconds)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Process single file instead of all"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already processed pairs"
    )

    args = parser.parse_args()

    output_dir = args.output or (args.input.parent / "tailored")

    if args.file:
        # Single file
        rel_path = args.file.name
        output_path = output_dir / rel_path
        print(f"Processing {args.file} with {args.provider}/{args.model}")
        stats = process_file(
            args.file, output_path,
            provider=args.provider,
            model=args.model,
            batch_size=args.batch_size,
            delay=args.delay,
            skip_existing=not args.no_skip
        )
        print(f"Done: {stats['processed']} tailored, {stats['skipped']} skipped")
    else:
        # All files
        total_stats = {"processed": 0, "skipped": 0, "total": 0}
        print(f"Using {args.provider}/{args.model}")

        for jsonl_file in sorted(args.input.rglob("*.jsonl")):
            rel_path = jsonl_file.relative_to(args.input)
            output_path = output_dir / rel_path

            print(f"\n{rel_path}")
            stats = process_file(
                jsonl_file, output_path,
                provider=args.provider,
                model=args.model,
                batch_size=args.batch_size,
                delay=args.delay,
                skip_existing=not args.no_skip
            )

            for k in total_stats:
                total_stats[k] += stats[k]

        print(f"\n=== Total ===")
        print(f"Processed: {total_stats['processed']}")
        print(f"Skipped: {total_stats['skipped']}")
        print(f"Total pairs: {total_stats['total']}")


if __name__ == "__main__":
    main()

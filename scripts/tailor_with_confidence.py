#!/usr/bin/env python3
"""
Tailor Answers with Confidence Scoring (Combined)

Generates tailored answers AND confidence scores in a single LLM call.
Half the prompts compared to separate tailoring + scoring passes.

Usage:
    python3 scripts/tailor_with_confidence.py --input datasets/skills_qa/skills_qa.jsonl
    python3 scripts/tailor_with_confidence.py --input datasets/skills_qa/skills_qa.jsonl --provider gemini
"""

import argparse
import json
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional

project_root = Path(__file__).parent.parent
SKILLS_DIR = project_root / "skills"


COMBINED_PROMPT = '''You are helping create training data for UnifyWeaver's semantic search system.

Given a question and base answer, provide:
1. A TAILORED answer that directly addresses the specific question
2. A CONFIDENCE score (0.0-1.0) for how well the base answer supports this question

SKILL DOCUMENT (for reference):
{skill_content}

QUESTION: {question}

BASE ANSWER:
{base_answer}

Rewrite the answer to directly address this specific question. Keep essential technical info.
Then rate your confidence that this answer correctly addresses the question.

Output ONLY valid JSON (no markdown, no explanation):
{{"answer": "your tailored answer here", "confidence": 0.85, "issues": null}}

If there are concerns, note them briefly in "issues". Otherwise set issues to null.'''


def load_skill_content(skill_file: str) -> str:
    """Load skill document content."""
    if not skill_file:
        return ""

    skill_path = SKILLS_DIR / skill_file
    if skill_path.exists():
        content = skill_path.read_text(encoding='utf-8')
        # Truncate long docs
        if len(content) > 3000:
            content = content[:3000] + "\n\n[... truncated ...]"
        return content

    # Try without .md
    if not skill_file.endswith('.md'):
        skill_path = SKILLS_DIR / f"{skill_file}.md"
        if skill_path.exists():
            content = skill_path.read_text(encoding='utf-8')
            if len(content) > 3000:
                content = content[:3000] + "\n\n[... truncated ...]"
            return content

    return ""


def call_llm(prompt: str, provider: str = "claude", model: str = "haiku", timeout: int = 90) -> Optional[str]:
    """Call LLM and return response."""
    try:
        if provider == "claude":
            result = subprocess.run(
                ["claude", "-p", "--model", model, prompt],
                capture_output=True, text=True, timeout=timeout
            )
        else:  # gemini
            result = subprocess.run(
                ["gemini", "-p", prompt, "-m", model, "--output-format", "text"],
                capture_output=True, text=True, timeout=timeout
            )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  LLM error: {e}", file=sys.stderr)
        return None


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling common issues."""
    if not response:
        return {"answer": None, "confidence": 0.5, "issues": "No response"}

    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    if "```" in response:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # Try finding JSON object in response
    match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: treat whole response as answer
    return {"answer": response[:2000], "confidence": 0.5, "issues": "Could not parse JSON"}


def process_pair(
    pair: Dict[str, Any],
    provider: str,
    model: str
) -> Dict[str, Any]:
    """Process a single Q/A pair: tailor and score confidence."""

    question = pair.get("question", "")
    base_answer = pair.get("answer", "")

    # Get skill content for context
    skill_files = pair.get("related_skills", [])
    skill_content = ""
    for sf in skill_files:
        content = load_skill_content(sf)
        if content:
            skill_content = content
            break

    if not skill_content:
        skill_content = "(No skill document available)"

    # Build prompt
    prompt = COMBINED_PROMPT.format(
        skill_content=skill_content,
        question=question,
        base_answer=base_answer[:2000]
    )

    # Call LLM
    response = call_llm(prompt, provider, model)
    result = parse_llm_response(response)

    # Build output record
    output = pair.copy()
    output["original_answer"] = base_answer

    if result.get("answer"):
        output["answer"] = result["answer"]
        output["answer_tailored"] = True
    else:
        output["answer_tailored"] = False

    output["confidence"] = result.get("confidence", 0.5)
    output["confidence_issues"] = result.get("issues")

    return output


def process_file(
    input_path: Path,
    output_path: Path,
    provider: str,
    model: str,
    batch_size: int = 10,
    delay: float = 0.5,
    skip_existing: bool = True
) -> Dict[str, int]:
    """Process a JSONL file."""

    # Load existing if skip_existing
    existing = {}
    if skip_existing and output_path.exists():
        with open(output_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if record.get("answer_tailored"):
                            existing[record.get("pair_id", "")] = record
                    except:
                        pass

    # Load input
    pairs = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    pairs.append(json.loads(line))
                except:
                    pass

    # Filter already processed
    to_process = []
    already_done = []
    for pair in pairs:
        pair_id = pair.get("pair_id", "")
        if pair_id in existing:
            already_done.append(existing[pair_id])
        else:
            to_process.append(pair)

    if not to_process:
        print(f"  All {len(already_done)} pairs already processed")
        return {"processed": 0, "skipped": len(already_done), "total": len(pairs)}

    print(f"  Processing {len(to_process)} pairs ({len(already_done)} already done)")

    # Process
    results = already_done.copy()

    for i, pair in enumerate(to_process):
        print(f"  [{i+1}/{len(to_process)}] {pair.get('question', '')[:50]}...")

        result = process_pair(pair, provider, model)
        results.append(result)

        conf = result.get('confidence', 0)
        status = "✓" if result.get('answer_tailored') else "✗"
        print(f"    {status} confidence: {conf:.2f}")

        # Save progress after each batch
        if (i + 1) % batch_size == 0 or i == len(to_process) - 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            print(f"    Saved {len(results)} pairs")

        if delay > 0 and i < len(to_process) - 1:
            time.sleep(delay)

    # Stats
    processed = sum(1 for r in results if r.get("answer_tailored"))
    avg_conf = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
    low_conf = sum(1 for r in results if r.get("confidence", 1) < 0.7)

    print(f"\n  Processed: {processed}/{len(results)}")
    print(f"  Avg confidence: {avg_conf:.3f}")
    print(f"  Low confidence (<0.7): {low_conf}")

    return {"processed": processed, "skipped": len(already_done), "total": len(pairs)}


def main():
    parser = argparse.ArgumentParser(
        description="Tailor answers with confidence scoring in one pass"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--provider", default="claude", choices=["claude", "gemini"])
    parser.add_argument("--model", default="haiku", help="Model name")
    parser.add_argument("--batch-size", type=int, default=10, help="Save frequency")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between calls")
    parser.add_argument("--no-skip", action="store_true", help="Reprocess all")

    args = parser.parse_args()

    output_dir = args.output or args.input.parent / "tailored_scored"
    output_path = output_dir / args.input.name

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    print(f"Provider: {args.provider}/{args.model}")
    print()

    stats = process_file(
        args.input,
        output_path,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        delay=args.delay,
        skip_existing=not args.no_skip
    )

    print(f"\nDone: {stats['processed']} tailored, {stats['skipped']} skipped")


if __name__ == "__main__":
    main()

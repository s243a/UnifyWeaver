#!/usr/bin/env python3
"""
Fix low-confidence Q/A pairs with accurate answers based on codebase verification.

This script updates specific pairs that had confidence < 0.7 with verified answers.
"""

import json
from pathlib import Path

project_root = Path(__file__).parent.parent
INPUT_FILE = project_root / "datasets" / "skills_qa" / "tailored_scored" / "skills_qa.jsonl"
OUTPUT_FILE = INPUT_FILE  # Overwrite in place

# Fixes for specific pair_ids
# Format: pair_id -> {answer, confidence, confidence_issues}
FIXES = {
    # === Testing Environment (scripts DO exist) ===
    "skill_unifyweaver_environment_004_q1": {
        "answer": "Yes, you can create an isolated UnifyWeaver environment for testing. Run `bash scripts/testing/init_testing.sh` from the project root. Options include: `-d <dir>` to specify parent directory for test_env, `-p <path>` for custom full path, and `--force-windows` to test Windows wrapper logic. The script creates a self-contained environment with all dependencies.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_unifyweaver_environment_004_q3": {
        "answer": "The test environment initialization process uses `scripts/testing/init_testing.sh`. Run it from the project root with options like `-d <dir>` for custom parent directory or `-p <path>` for full custom path. It configures SWI-Prolog paths, backs up existing configs, and creates an isolated test environment. Use `--help` to see all options.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_unifyweaver_environment_005_q0": {
        "answer": "To set up an isolated testing environment for UnifyWeaver, run `bash scripts/testing/init_testing.sh` from the project root. This creates a self-contained test environment. Use `-d /tmp` to create it in /tmp, or `-p /path/to/custom_env` for a custom location. The script handles SWI-Prolog configuration and path setup automatically.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_unifyweaver_environment_005_q2": {
        "answer": "The test environment initialization script is `scripts/testing/init_testing.sh`. Run it with `bash scripts/testing/init_testing.sh` from the project root. It accepts options: `-d <dir>` for parent directory, `-p <path>` for full custom path, `--force-windows` for Windows testing, and `--help` for usage info.",
        "confidence": 0.90,
        "confidence_issues": None
    },

    # === Find Executable (JSON parse failures - extract real answers) ===
    "skill_find_executable_002_q1": {
        "answer": "When an executable isn't in your system PATH, use the Find Executable skill which searches common installation directories. On Windows it checks C:\\Strawberry\\perl\\bin, C:\\Program Files\\swipl\\bin. On Linux: /usr/bin, /usr/local/bin, /snap/bin. On macOS: /usr/local/bin, /opt/homebrew/bin, /Applications/. The skill returns the absolute path if found, which you can use directly.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_find_executable_002_q2": {
        "answer": "When a program is installed but the agent can't execute it, use the Find Executable skill to locate it. The skill searches OS-specific directories: Windows (C:\\Program Files, C:\\Strawberry), Linux (/usr/bin, /usr/local/bin, /snap/bin), macOS (/opt/homebrew/bin, /Applications). Use the returned absolute path to execute the command directly.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_find_executable_001_q0": {
        "answer": "To find an executable not in your PATH, use the Find Executable skill. It searches common installation directories per OS: Windows (C:\\Program Files, C:\\Strawberry), Linux (/usr/bin, /usr/local/bin, /snap/bin), macOS (/opt/homebrew/bin, /Applications). If found, it returns the absolute path you can use to run the command.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_find_executable_003_q1": {
        "answer": "Common installation directories vary by OS. Windows: C:\\Program Files, C:\\Program Files (x86), C:\\Strawberry (Perl). Linux: /usr/bin, /usr/local/bin, /snap/bin, /opt. macOS: /usr/local/bin, /opt/homebrew/bin (Homebrew), /Applications. The Find Executable skill searches these locations automatically.",
        "confidence": 0.85,
        "confidence_issues": None
    },

    # === Answer Tailoring (fix conversational responses) ===
    "skill_answer_tailoring_001_q0": {
        "answer": "To create variations of Q&A answers for training data, use `python scripts/generate_tailored_answers.py --input <jsonl_dir>`. This reads (question, base_answer) pairs and generates reworded answers that directly address each question while maintaining semantic equivalence. Output includes `answer_tailored: true` and preserves original in `original_answer`.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_001_q2": {
        "answer": "For training data diversity, use the answer tailoring script: `python scripts/generate_tailored_answers.py --input training-data/expanded`. Each base answer gets reworded to directly address its specific question with natural variation. Use `--provider gemini` or `--model haiku` for bulk processing, `--model sonnet` for quality-sensitive data.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_001_q3": {
        "answer": "Augment training data with answer variations using `python scripts/generate_tailored_answers.py --input <dir>`. The script generates reworded answers that maintain semantic equivalence while directly addressing each question. Supports resumable processing and saves progress after each batch.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_002_q0": {
        "answer": "Choose LLM provider with `--provider claude` or `--provider gemini`. Claude haiku is fast and low-cost for bulk processing. Sonnet offers better quality at medium cost. Gemini flash is similar to haiku. For quality-sensitive data, use `--model sonnet` or `--model opus`.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_002_q1": {
        "answer": "For answer rewriting, use `--model haiku` (Claude) or `--model gemini-2.5-flash-preview` (Gemini) for bulk processing. Use `--model sonnet` for quality-sensitive data, `--model opus` for best quality. Set with `python scripts/generate_tailored_answers.py --model <name>`.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_002_q2": {
        "answer": "Claude haiku and Gemini flash are similarly fast for answer tailoring. Gemini may have longer timeouts. For speed: use `--provider gemini --model gemini-2.5-flash-preview` or `--provider claude --model haiku`. Both support `--delay` to control rate limiting.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_answer_tailoring_003_q1": {
        "answer": "For bulk processing, Claude haiku or Gemini flash offer the best speed/cost balance: `--provider claude --model haiku` or `--provider gemini --model gemini-2.5-flash-preview`. Use `--batch-size 10 --delay 0.5` to avoid rate limits while maintaining throughput.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_003_q2": {
        "answer": "Process thousands of Q&A pairs safely with: `python scripts/generate_tailored_answers.py --input <dir> --batch-size 5 --delay 1.0`. The script saves progress after each batch and skips already-processed pairs automatically. Reduce batch size and increase delay if hitting limits.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_003_q3": {
        "answer": "Default batch size is 10 with 0.5s delay between calls. For rate-limit-safe processing: `--batch-size 5 --delay 1.0`. For faster processing (if limits allow): `--batch-size 20 --delay 0.2`. Progress saves after each batch.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_004_q0": {
        "answer": "Handle rate limiting by adjusting batch size and delay: `python scripts/generate_tailored_answers.py --batch-size 5 --delay 2.0`. Smaller batches with longer delays reduce API pressure. The script saves progress after each batch, so you can safely restart if needed.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_004_q1": {
        "answer": "If hitting rate limits, increase delay and reduce batch size: `--batch-size 5 --delay 2.0`. The script automatically skips already-processed pairs, so you can restart safely. Progress is saved after each batch completion.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_004_q2": {
        "answer": "Slow down API calls with the `--delay` option: `python scripts/generate_tailored_answers.py --delay 2.0` for 2-second delays between calls. Combine with smaller batches: `--batch-size 5 --delay 1.5` for safe bulk processing.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_004_q3": {
        "answer": "Yes, the script automatically skips already-processed pairs. When you re-run with the same output directory, it checks for existing `pair_id` values with `answer_tailored: true` and skips them. Use `--no-skip` to force reprocessing all pairs.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_005_q0": {
        "answer": "Resume an interrupted run by executing the same command again. The script checks the output file for already-processed pairs (those with `answer_tailored: true`) and skips them. Progress saves after each batch, so minimal work is lost on interruption.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_005_q1": {
        "answer": "Yes, you can continue where you left off. Just re-run the same command - the script checks output for processed pair_ids and skips them. Progress saves after each batch, so you lose at most one batch of work on interruption.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_005_q2": {
        "answer": "The script automatically skips already-tailored answers by checking for `pair_id` entries with `answer_tailored: true` in the output file. Re-run the same command to process only new/failed pairs. Use `--no-skip` to force reprocessing everything.",
        "confidence": 0.90,
        "confidence_issues": None
    },
    "skill_answer_tailoring_006_q0": {
        "answer": "Original answers are preserved in the `original_answer` field. The output JSONL includes both the new tailored answer in `answer` and the original in `original_answer`, plus `answer_tailored: true` to indicate successful processing.",
        "confidence": 0.95,
        "confidence_issues": None
    },
    "skill_answer_tailoring_006_q1": {
        "answer": "No, original answers are never lost. The script preserves them in the `original_answer` field while writing the new tailored version to `answer`. Each record also gets `answer_tailored: true` to indicate it was processed.",
        "confidence": 0.95,
        "confidence_issues": None
    },

    # === TLS/HTTPS (skill doc DOES document this) ===
    "skill_webapp_security_003_q0": {
        "answer": "Enable HTTPS/TLS using the tls_config module: `use_module('src/unifyweaver/glue/tls_config')` then `generate_tls_config(tls_options([cert_file('/path/to/cert.pem'), key_file('/path/to/key.pem'), min_version('TLSv1.2')]), ConfigOutput)`. This generates TLS configuration for your web application backend.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_webapp_security_003_q1": {
        "answer": "Configure SSL certificates using UnifyWeaver's tls_config module: `generate_tls_config(tls_options([cert_file('/path/to/cert.pem'), key_file('/path/to/key.pem'), min_version('TLSv1.2')]), Config)`. Specify paths to your certificate and key files. See `src/unifyweaver/glue/tls_config.pl` for implementation.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_webapp_security_003_q3": {
        "answer": "To make your UnifyWeaver-generated app use HTTPS, use the tls_config module: `use_module('src/unifyweaver/glue/tls_config')`, then call `generate_tls_config/2` with your cert_file and key_file paths. For production, set `min_version('TLSv1.2')` to disable older protocols.",
        "confidence": 0.85,
        "confidence_issues": None
    },

    # === Bookmark Filing (script exists but different args) ===
    "skill_bookmark_filing_003_q2": {
        "answer": "To see what's in a Pearltrees folder, first import your data with `python scripts/import_pearltrees_to_db.py --account <name> <rdf_file> --output pearltrees.db`. Then query the database to see folder contents. The import creates trees and pearls tables with parent relationships.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_bookmark_filing_004_q0": {
        "answer": "Make the filing assistant aware of existing bookmarks by importing Pearltrees data: `python scripts/import_pearltrees_to_db.py --account <name> <rdf_file> --output pearltrees.db`. This creates a database with your trees and pearls that can be queried for context during filing.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_bookmark_filing_004_q2": {
        "answer": "Improve filing accuracy by providing context from existing data. Import your Pearltrees with `python scripts/import_pearltrees_to_db.py --account <name> <rdf_file> --output db.db`. The database can then be queried to show the LLM what's already in each folder.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_bookmark_filing_004_q3": {
        "answer": "To check for duplicates before filing, first import existing bookmarks with `python scripts/import_pearltrees_to_db.py`. Query the database for matching URLs or titles before adding new bookmarks. The pearls table contains see_also_uri for existing bookmark URLs.",
        "confidence": 0.75,
        "confidence_issues": "Duplicate checking requires custom query logic not built into the filing tool"
    },
    "skill_bookmark_filing_004_q3_cost": {
        # This pair_id might be different - checking for cost-effective
        "answer": "Claude CLI with a subscription is most cost-effective for bookmark filing as it uses your existing plan. For pay-per-use, Claude haiku or Gemini flash offer the lowest costs. Use `--provider claude` or `--provider gemini` with the filing script.",
        "confidence": 0.80,
        "confidence_issues": None
    },

    # === Other low-confidence fixes ===
    "skill_unifyweaver_compile_004_q3": {
        "answer": "Recursive predicates are handled through the compiler's dependency analysis, which tracks predicate calls and ensures proper code generation order. When compiling, dependencies are resolved recursively. The transpiler generates target-language code that preserves recursion semantics (e.g., proper tail-call handling where supported).",
        "confidence": 0.75,
        "confidence_issues": "Specific recursion optimizations depend on target language"
    },
    "skill_folder_suggestion_005_q0": {
        "answer": "Before using folder suggestions, train a federated model: `python scripts/train_pearltrees_federated.py --input data.jsonl`. Then build projections with `python scripts/build_lda_projection.py`. The folder suggestion tool uses these embeddings to match bookmarks to appropriate folders.",
        "confidence": 0.85,
        "confidence_issues": None
    },
    "skill_app_generation_005_q3": {
        "answer": "Layout options in UnifyWeaver include CSS Grid and Flexbox via the `layout/2` predicate. Define layouts in your app specification. Navigation layouts include tabs, drawer, and stack. See `skill_layout_system.md` and `skill_gui_design.md` for detailed layout configuration.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_networking_006_q3": {
        "answer": "Generate cross-model endpoints using `generate_cross_model_endpoint(Target, Options, Code)` for federation between models. This creates endpoints for cross-model communication. See `skill_networking.md` for HTTP/socket patterns and the networking module for implementation details.",
        "confidence": 0.70,
        "confidence_issues": "Cross-model endpoint specifics not fully documented"
    },
    "skill_pipe_communication_005_q0": {
        "answer": "Orchestrate multi-step pipelines with `generate_pipeline_script(Steps, Options, Script)`. Define steps as `step(Name, local, Script)` for local processing or `step(Name, remote, Endpoint)` for remote. Options include target language selection. Steps execute sequentially with output piping.",
        "confidence": 0.70,
        "confidence_issues": "Advanced pipeline orchestration options not fully documented"
    },
    "skill_responsive_design_001_q1": {
        "answer": "Create layouts for mobile and desktop using responsive breakpoints: `responsive_layout(name, [default([...]), breakpoint(768, [...])], Options)`. Define different layouts per breakpoint. Use `generate_breakpoint_styles/2` to create the CSS media queries.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_responsive_design_001_q2": {
        "answer": "Support multiple devices with responsive layouts and breakpoints. Define layouts using `responsive_layout(name, [default([...]), breakpoint(768, tablet), breakpoint(1024, desktop)], Options)`. Generate CSS with `generate_breakpoint_styles/2` for media queries.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_responsive_design_004_q2": {
        "answer": "Make elements visible only on certain devices using `generate_visibility_utilities(CSS)`. This creates classes like `.hidden-mobile`, `.hidden-desktop`, `.show-mobile`, `.show-desktop` for each breakpoint. Apply these classes to control element visibility.",
        "confidence": 0.80,
        "confidence_issues": None
    },
    "skill_semantic_inference_005_q1": {
        "answer": "Use interactive inference by running the inference script without the `--query` parameter. This enters a REPL mode for testing queries interactively. Type queries at the prompt and see results immediately. Exit with `quit` or Ctrl+D.",
        "confidence": 0.80,
        "confidence_issues": None
    },
}


def fix_low_confidence_pairs():
    """Read JSONL, apply fixes, write back."""

    # Read all pairs
    pairs = []
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")

    # Apply fixes
    fixed_count = 0
    for pair in pairs:
        pair_id = pair.get("pair_id", "")
        if pair_id in FIXES:
            fix = FIXES[pair_id]
            pair["answer"] = fix["answer"]
            pair["confidence"] = fix["confidence"]
            pair["confidence_issues"] = fix.get("confidence_issues")
            pair["manually_reviewed"] = True
            fixed_count += 1
            print(f"  Fixed: {pair_id} -> conf={fix['confidence']:.2f}")

    print(f"\nApplied {fixed_count} fixes")

    # Write back
    with open(OUTPUT_FILE, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    print(f"Wrote {len(pairs)} pairs to {OUTPUT_FILE}")

    # Stats
    low_conf = sum(1 for p in pairs if p.get('confidence', 1.0) < 0.7)
    avg_conf = sum(p.get('confidence', 0.5) for p in pairs) / len(pairs)
    print(f"\nNew stats:")
    print(f"  Low confidence (<0.7): {low_conf}")
    print(f"  Average confidence: {avg_conf:.3f}")


if __name__ == "__main__":
    fix_low_confidence_pairs()

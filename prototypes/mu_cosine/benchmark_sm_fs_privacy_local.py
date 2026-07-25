#!/usr/bin/env python3
"""Synthetic-only contract check for a local SimpleMind privacy reviewer.

This benchmark never reads the SimpleMind corpus.  Passing it is necessary but
not sufficient for an advisory pilot: a separate, versioned benchmark lock and
owner-authorized pilot protocol would still be required.  Model recommendations
remain ``unknown`` in the privacy index implemented here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from llm_cli import (  # noqa: E402
    call_ollama_json,
    local_ollama_endpoint,
    resolve_ollama_cli,
)
from sm_fs_privacy import (  # noqa: E402
    INTERPRETATIONS,
    MODEL_RESPONSE_SCHEMA,
    POLICY_ID,
    SmFsPrivacyError,
    _parse_model_response,
    build_review_prompt,
    canonical_json_bytes,
    sha256_file,
)


SCHEMA = "unifyweaver.sm-fs-privacy-synthetic-benchmark.v2"
REVIEW_ELIGIBLE = frozenset(
    ("AC-04", "TP-02", "TP-03", "UN-01", "UN-02", "UN-03", "UN-04")
)


def _case(
    case_id: str,
    relative_path: str,
    root_title: str,
    root_urls: Sequence[str],
    rule_reasons: Sequence[str],
    expected: str,
) -> Dict[str, Any]:
    qid = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
    return {
        "case_id": case_id,
        "expected": expected,
        "review_eligible": case_id in REVIEW_ELIGIBLE,
        "task": {
            "qid": qid,
            "relative_path": relative_path,
            "root_title": root_title,
            "root_urls": list(root_urls),
            "rule_reasons": list(rule_reasons),
            "allowed_interpretations": sorted(INTERPRETATIONS),
        },
    }


def synthetic_cases() -> List[Dict[str, Any]]:
    return [
        _case(
            "AC-01",
            "Synthetic/Private/Field Notes.smmx",
            "Field Notes",
            (),
            ("private_dir_or_access_prefix",),
            "access_control",
        ),
        _case(
            "AC-02",
            "Synthetic/Private_Project Atlas.smmx",
            "Private: Project Atlas",
            (),
            ("private_dir_or_access_prefix",),
            "access_control",
        ),
        _case(
            "AC-03",
            "Synthetic/Archive/Private - Team Copy.smmx",
            "Team Copy",
            ("https://www.pearltrees.com/private/id1001",),
            ("private_dir_or_access_prefix", "pearltrees_private_link_unknown"),
            "access_control",
        ),
        _case(
            "AC-04",
            "Synthetic/Archive/Private copy of Team Notes.smmx",
            "Copy of Team Notes",
            ("https://www.pearltrees.com/private/id1002",),
            ("private_word_marker_vs_topic", "pearltrees_private_link_unknown"),
            "access_control",
        ),
        _case(
            "TP-01",
            "Synthetic/STEM/Private-key cryptography.smmx",
            "Private-key cryptography",
            ("https://www.pearltrees.com/example/private-key/id2001",),
            ("public_root_link",),
            "topical",
        ),
        _case(
            "TP-02",
            "Synthetic/Law/Privacy law.smmx",
            "Privacy law",
            ("https://en.wikipedia.org/wiki/Privacy_law",),
            ("private_word_marker_vs_topic",),
            "topical",
        ),
        _case(
            "TP-03",
            "Synthetic/Statistics/MMSE.smmx",
            "Minimum mean square error",
            ("https://www.pearltrees.com/private/id2003",),
            ("pearltrees_private_link_unknown",),
            "topical",
        ),
        _case(
            "TP-04",
            "Synthetic/Programming/C++ private methods.smmx",
            "C++ private methods",
            ("https://www.pearltrees.com/example/cpp-private/id2004",),
            ("public_root_link",),
            "topical",
        ),
        _case(
            "UN-01",
            "Synthetic/Private/Private-key cryptography.smmx",
            "Private-key cryptography",
            (),
            ("private_marker_with_topical_root",),
            "uncertain",
        ),
        _case(
            "UN-02",
            "Synthetic/Topics/Private research.smmx",
            "Private research",
            ("https://www.pearltrees.com/private/id3002",),
            ("private_word_marker_vs_topic", "pearltrees_private_link_unknown"),
            "uncertain",
        ),
        _case(
            "UN-03",
            "Synthetic/Topics/Private networks.smmx",
            "Private networks",
            (),
            ("private_word_marker_vs_topic",),
            "uncertain",
        ),
        _case(
            "UN-04",
            "Synthetic/Topics/Private.smmx",
            "Private",
            (),
            ("private_word_marker_vs_topic",),
            "uncertain",
        ),
    ]


def score_response(
    response: Optional[str], cases: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    tasks = [case["task"] for case in cases]
    expected_qids = {task["qid"] for task in tasks}
    result = {
        "transport_ok": response is not None,
        "raw_unfenced_json": False,
        "schema_ok": False,
        "labels": {},
        "correct": 0,
        "review_eligible_correct": 0,
        "raw_response": response,
        "raw_response_sha256": (
            hashlib.sha256(response.encode("utf-8")).hexdigest()
            if response is not None
            else None
        ),
        "error": None,
    }
    if response is None:
        result["error"] = "model call failed"
        return result
    stripped = response.strip()
    result["raw_unfenced_json"] = stripped.startswith("[") and stripped.endswith(
        "]"
    ) and "```" not in stripped
    try:
        decisions = _parse_model_response(response, expected_qids)
    except SmFsPrivacyError as exc:
        result["error"] = str(exc)
        return result
    result["schema_ok"] = True
    expected_by_case = {case["case_id"]: case["expected"] for case in cases}
    for case in cases:
        predicted = decisions[case["task"]["qid"]]["interpretation"]
        result["labels"][case["case_id"]] = predicted
        if predicted == expected_by_case[case["case_id"]]:
            result["correct"] += 1
            if case["review_eligible"]:
                result["review_eligible_correct"] += 1
    return result


def _command_text(command: Sequence[str], timeout: int = 30) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _model_identity(model: str) -> Dict[str, str]:
    cli = resolve_ollama_cli()
    listing = _command_text((cli, "list"))
    digest = ""
    for line in listing.splitlines()[1:]:
        columns = line.split()
        if len(columns) >= 2 and columns[0] == model:
            digest = columns[1]
            break
    return {
        "tag": model,
        "digest": digest,
        "ollama_version": _command_text((cli, "--version")),
        "ollama_cli": cli,
    }


def _provenance_snapshot(model: str) -> Dict[str, Any]:
    """Capture the code, repository, endpoint, and resolved model before a run."""

    model_identity = _model_identity(model)
    if (
        not re.fullmatch(r"[0-9a-fA-F]{12,64}", model_identity["digest"])
        or not model_identity["ollama_version"]
        or not model_identity["ollama_cli"]
    ):
        raise RuntimeError("cannot bind the installed Ollama model identity")
    repo_commit = _command_text(("git", "-C", str(ROOT), "rev-parse", "HEAD"))
    if not re.fullmatch(r"[0-9a-f]{40}", repo_commit):
        raise RuntimeError("cannot bind the repository commit")
    return {
        "model": model_identity,
        "repo_commit": repo_commit,
        "endpoint": local_ollama_endpoint(),
        "sources": {
            "classifier_sha256": sha256_file(HERE / "sm_fs_privacy.py"),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "llm_bridge_sha256": sha256_file(ROOT / "scripts" / "llm_cli.py"),
        },
    }


def _install_private_json(path: Path, value: Mapping[str, Any]) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        os.chmod(path.parent, 0o700)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False
    ).encode("utf-8") + b"\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="explicit shared output budget; never rely on a model-specific default",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="enable the model's reasoning channel while scoring only its final JSON",
    )
    parser.add_argument(
        "--out",
        default=os.path.expanduser(
            "~/mu_data/sm_fs_privacy_qwen3_4b_synthetic.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")

    provenance = _provenance_snapshot(args.model)
    cases = synthetic_cases()
    tasks = [case["task"] for case in cases]
    prompt = build_review_prompt(tasks)
    response_schema = copy.deepcopy(MODEL_RESPONSE_SCHEMA)
    response_schema["minItems"] = len(tasks)
    response_schema["maxItems"] = len(tasks)
    started = datetime.now(timezone.utc)
    runs = []
    for repetition in range(args.repetitions):
        run_started = datetime.now(timezone.utc)
        clock = time.monotonic()
        ollama_eval: Dict[str, Any] = {}
        response = call_ollama_json(
            prompt,
            args.model,
            args.timeout,
            response_schema=response_schema,
            max_output_tokens=args.max_output_tokens,
            think=args.reasoning,
            metadata_out=ollama_eval,
        )
        scored = score_response(response, cases)
        scored.update(
            {
                "repetition": repetition,
                "started_at": run_started.isoformat(),
                "elapsed_seconds": round(time.monotonic() - clock, 6),
                "ollama_eval": ollama_eval,
            }
        )
        runs.append(scored)

    if _provenance_snapshot(args.model) != provenance:
        raise RuntimeError("benchmark provenance changed while the model was evaluated")

    label_vectors = [
        tuple(run["labels"].get(case["case_id"]) for case in cases)
        for run in runs
    ]
    strict_pass = bool(runs) and all(
        run["transport_ok"]
        and run["raw_unfenced_json"]
        and run["schema_ok"]
        and run["correct"] == len(cases)
        for run in runs
    ) and len(set(label_vectors)) == 1
    confusion = Counter()
    for run in runs:
        for case in cases:
            prediction = run["labels"].get(case["case_id"], "missing")
            confusion[(case["expected"], prediction)] += 1
    report = {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "private_data_used": False,
        "decision": (
            "benchmark_gate_passed_lock_not_implemented"
            if strict_pass
            else "advisory_pilot_not_allowed"
        ),
        "strict_pass": strict_pass,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "provider": "ollama-json",
        "endpoint": provenance["endpoint"],
        "decoding": {
            "stream": False,
            "format": "json_schema",
            "schema_sha256": hashlib.sha256(
                canonical_json_bytes(response_schema)
            ).hexdigest(),
            "think": args.reasoning,
            "temperature": 0,
            "seed": 0,
            "num_predict": args.max_output_tokens,
        },
        "model": provenance["model"],
        "repo_commit": provenance["repo_commit"],
        "case_set_sha256": hashlib.sha256(
            b"".join(canonical_json_bytes(case) for case in cases)
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        **provenance["sources"],
        "cases": cases,
        "runs": runs,
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
        "review_eligible_total_per_run": sum(
            case["review_eligible"] for case in cases
        ),
        "note": (
            "Synthetic contract evidence only. Passing never authorizes model "
            "recommendations to replace owner classifications."
        ),
    }
    _install_private_json(Path(args.out), report)
    print(
        f"synthetic privacy benchmark: {'PASS' if strict_pass else 'FAIL'}; "
        f"runs={len(runs)}; output={args.out}"
    )
    return 0 if strict_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

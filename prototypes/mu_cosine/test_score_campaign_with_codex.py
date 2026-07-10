#!/usr/bin/env python3
"""Tests for resumable campaign scoring checkpoints."""

import json
import re
import tempfile
from pathlib import Path

from score_campaign_with_codex import CampaignScoringError, build_parser, run


def write_pairs(path, count=3):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# node_title\troot_title\tcur_relation\tconf\tneighborhood\tnode_type\troot_type\traw\n")
        for index in range(count):
            f.write(
                f"Child {index}\tParent {index}\tsubtopic\t1.0\tprincipal_h1\t"
                "mindmap_node\tmindmap_node\t\n"
            )


def response_object(index):
    return {
        "id": index,
        "element_of": {"mu_fwd": 0.1, "mu_rev": 0.0, "applies": 0.1},
        "subcategory": {"mu_fwd": 0.7, "mu_rev": 0.1, "applies": 0.7},
        "subtopic": {"mu_fwd": 0.8, "mu_rev": 0.1, "applies": 0.9},
        "super_category": {"mu_fwd": 0.1, "mu_rev": 0.7, "applies": 0.1},
        "see_also": {"mu": 0.2, "applies": 0.2},
        "assoc": {"mu": 0.3, "applies": 0.3},
        "none": {"applies": 0.1},
        "unknown": {"mu_fwd": 0.1, "mu_rev": 0.1, "applies": 0.1},
    }


def fake_caller(prompt, _model, _effort, _sandbox, _timeout):
    ids = [int(value) for value in re.findall(r"^\s*(\d+)\. NODE:", prompt, re.MULTILINE)]
    return "judge output\n" + json.dumps([response_object(index) for index in ids])


def args_for(root, pairs, *extra):
    args = [
        "--pairs", str(pairs), "--batch", "2",
        "--checkpoint-dir", str(root / "checkpoints"),
        "--responses", str(root / "responses.txt"),
        "--out", str(root / "scored.tsv"),
        "--manifest", str(root / "manifest.json"),
    ]
    args.extend(extra)
    return build_parser().parse_args(args)


def test_scores_then_resumes_with_byte_stable_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        write_pairs(pairs)
        calls = []

        def counting_caller(*args):
            calls.append(args[0])
            return fake_caller(*args)

        args = args_for(root, pairs)
        assert run(args, caller=counting_caller) == 0
        assert len(calls) == 2
        assert len(list((root / "checkpoints").glob("batch_*.json"))) == 2
        scored = root / "scored.tsv"
        assert sum(1 for line in scored.read_text(encoding="utf-8").splitlines() if not line.startswith("#")) == 3
        first = (
            (root / "responses.txt").read_bytes(),
            scored.read_bytes(),
            (root / "manifest.json").read_bytes(),
        )

        def must_not_call(*_args):
            raise AssertionError("resume unexpectedly called the judge")

        assert run(args, caller=must_not_call) == 0
        assert (root / "responses.txt").read_bytes() == first[0]
        assert scored.read_bytes() == first[1]
        assert (root / "manifest.json").read_bytes() == first[2]


def test_retries_bad_ids_without_checkpointing_failure():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        write_pairs(pairs, count=1)
        calls = []

        def flaky(prompt, *args):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps([response_object(99)])
            return fake_caller(prompt, *args)

        assert run(args_for(root, pairs, "--retries", "1"), caller=flaky) == 0
        assert len(calls) == 2
        checkpoint = json.loads((root / "checkpoints" / "batch_0000.json").read_text(encoding="utf-8"))
        assert [item["id"] for item in checkpoint["response"]] == [0]


def test_retries_non_integer_json_id():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        write_pairs(pairs, count=1)
        calls = []

        def fractional_then_valid(prompt, *args):
            calls.append(prompt)
            item = response_object(0)
            if len(calls) == 1:
                item["id"] = 0.5
            return json.dumps([item])

        assert run(args_for(root, pairs, "--retries", "1"), caller=fractional_then_valid) == 0
        assert len(calls) == 2


def test_rejects_mislabeled_judge_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        write_pairs(pairs, count=1)
        args = args_for(root, pairs, "--model", "gpt-5.6")
        try:
            run(args, caller=fake_caller)
        except CampaignScoringError as exc:
            assert "requires model=gpt-5.5" in str(exc)
        else:
            raise AssertionError("expected judge/model contract rejection")


def test_retries_incomplete_and_out_of_range_schema():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pairs = root / "pairs.tsv"
        write_pairs(pairs, count=1)
        calls = []

        def incomplete_then_valid(prompt, *args):
            calls.append(prompt)
            item = response_object(0)
            if len(calls) == 1:
                del item["unknown"]
            elif len(calls) == 2:
                item["assoc"]["mu"] = 1.2
            return json.dumps([item])

        assert run(args_for(root, pairs, "--retries", "2"), caller=incomplete_then_valid) == 0
        assert len(calls) == 3


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"all {len(tests)} campaign scoring tests passed")

#!/usr/bin/env python3
"""Focused tests for the two-process Pearltrees snapshot consensus gate."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import stat

import pytest

import declare_pearltrees_diffusion_sources as declaration
import prepare_pearltrees_diffusion_consensus as consensus


RESOURCE_CEILING = 1_000_000
PHYSICAL_RELATIONS = {
    "alias": True,
    "collection": True,
    "cross_link": False,
    "path": True,
    "ref": True,
    "shortcut": True,
}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _public_rdf() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:pt="https://www.pearltrees.com/rdf/0.1/#">
  <pt:Tree rdf:about="https://www.pearltrees.com/synthetic/id1">
    <pt:treeId>1</pt:treeId><pt:title>Public root</pt:title><pt:privacy>0</pt:privacy>
  </pt:Tree>
  <pt:Tree rdf:about="https://www.pearltrees.com/synthetic/id2">
    <pt:treeId>2</pt:treeId><pt:title>Public child</pt:title><pt:privacy>0</pt:privacy>
  </pt:Tree>
  <pt:RefPearl>
    <pt:parentTree rdf:resource="https://www.pearltrees.com/synthetic/id1" />
    <pt:seeAlso rdf:resource="https://www.pearltrees.com/synthetic/id2" />
    <pt:title>Public child</pt:title><pt:privacy>0</pt:privacy>
  </pt:RefPearl>
</rdf:RDF>
"""


def _case(tmp_path: Path, *, with_legacy: bool = False) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    local_root = tmp_path / "local"
    inputs.mkdir()
    local_root.mkdir()
    rdf_path = inputs / "evidence.rdf"
    rdf_path.write_bytes(_public_rdf())
    legacy_path = inputs / "legacy.tsv"
    if with_legacy:
        legacy_path.write_bytes(b"")
    policy_path = inputs / "policy.json"
    declaration_dir = local_root / "declaration"
    spec = declaration.build_source_spec(
        snapshot_label="synthetic-consensus-test",
        rdf=(("explicit-rdf", "synthetic", str(rdf_path)),),
        legacy_dag=str(legacy_path) if with_legacy else None,
    )
    declaration.install_source_spec(
        spec,
        output_dir=declaration_dir,
        local_root=local_root,
    )
    _write_json(
        policy_path,
        {
            "physical_edges": PHYSICAL_RELATIONS,
            "schema": "pearltrees-diffusion-relation-policy-v1",
        },
    )
    return {
        "attempt_a": local_root / "attempt-a",
        "attempt_b": local_root / "attempt-b",
        "local_root": local_root,
        "policy": policy_path,
        "receipt": local_root / "consensus",
        "rdf": rdf_path,
        "legacy": legacy_path,
        "source_spec": declaration_dir / declaration.SPEC_FILENAME,
    }


def _prepare(case: dict[str, Path], *, minimum_anchors: int = 1):
    return consensus.prepare_consensus(
        case["source_spec"],
        case["policy"],
        case["attempt_a"],
        case["attempt_b"],
        case["receipt"],
        case["local_root"],
        minimum_anchors=minimum_anchors,
        resource_ceiling_bytes=RESOURCE_CEILING,
    )


def _verify(case: dict[str, Path]):
    return consensus.verify_consensus_receipt(
        case["receipt"],
        attempt_a_dir=case["attempt_a"],
        attempt_b_dir=case["attempt_b"],
        source_spec=case["source_spec"],
        relation_policy=case["policy"],
    )


def test_exact_ready_consensus_runs_two_compilers_and_never_pools(tmp_path: Path) -> None:
    case = _case(tmp_path)

    receipt, exit_code = _prepare(case)

    assert exit_code == 0
    assert receipt["accepted"] is True
    assert receipt["reason"] == "exact_consensus_ready"
    assert receipt["canonical_attempt"] == "attempt_a"
    assert receipt["graph_observation_count"] == 1
    assert receipt["repeatability_verified"] is True
    assert receipt["graph_gate_pass"] is True
    assert receipt["no_pooling"] is True
    assert receipt["readiness"] == {
        "graph_asset_ready": True,
        "minimum_anchor_count": 1,
        "minimum_anchor_coverage_pass": True,
        "privacy_certified": True,
        "resource_ceiling_bytes": RESOURCE_CEILING,
    }
    assert all(receipt["comparison_checks"].values())
    assert [row["prepare_exit_code"] for row in receipt["attempts"]] == [0, 0]
    assert [row["verify_exit_code"] for row in receipt["attempts"]] == [0, 0]
    assert receipt["attempts"][0]["manifest_record"] == receipt["attempts"][1][
        "manifest_record"
    ]
    assert receipt["attempts"][0]["snapshot_fingerprint"] == receipt[
        "common_snapshot_fingerprint"
    ]
    assert receipt["attempts"][0]["observed_contract_bytes"] == receipt["attempts"][
        1
    ]["observed_contract_bytes"]
    assert receipt["common_records"]["fingerprint_core_sha256"] == receipt[
        "common_snapshot_fingerprint"
    ]
    assert set(receipt["input_records"]) == {"relation_policy", "source_spec"}
    assert set(receipt["implementation_records"]) == {
        "consensus_preparer",
        "runtime_identity",
        "snapshot_preparer_entrypoint",
        "source_declaration_validator",
    }
    assert len(receipt["repeatability_contract_sha256"]) == 64
    assert case["attempt_a"].is_dir() and case["attempt_b"].is_dir()
    assert stat.S_IMODE(case["receipt"].stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in case["receipt"].iterdir()
    )
    assert _verify(case) == receipt
    receipt_text = (case["receipt"] / consensus.RECEIPT_NAME).read_text(encoding="utf-8")
    assert str(tmp_path) not in receipt_text
    assert "Public root" not in receipt_text
    assert "explicit-rdf" not in receipt_text


def test_exact_but_undercovered_runs_are_recorded_as_scientifically_blocked(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    receipt, exit_code = _prepare(case, minimum_anchors=3)

    assert exit_code == 2
    assert receipt["accepted"] is False
    assert receipt["repeatability_verified"] is True
    assert receipt["graph_gate_pass"] is False
    assert receipt["reason"] == "graph_asset_not_ready"
    assert receipt["common_snapshot_fingerprint"] is not None
    assert receipt["readiness"]["privacy_certified"] is True
    assert receipt["readiness"]["minimum_anchor_coverage_pass"] is False
    assert receipt["readiness"]["graph_asset_ready"] is False
    assert [row["prepare_exit_code"] for row in receipt["attempts"]] == [2, 2]
    assert all(receipt["comparison_checks"].values())


def test_privacy_block_takes_reason_precedence_over_undercoverage(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["rdf"].write_bytes(
        _public_rdf().replace(b"<pt:privacy>0</pt:privacy>", b"")
    )

    receipt, exit_code = _prepare(case, minimum_anchors=3)

    assert exit_code == 2
    assert receipt["repeatability_verified"] is True
    assert receipt["readiness"]["privacy_certified"] is False
    assert receipt["readiness"]["minimum_anchor_coverage_pass"] is False
    assert receipt["reason"] == "privacy_not_certified"


def test_handwritten_spec_without_rdf_account_is_rejected_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    value = json.loads(case["source_spec"].read_text(encoding="utf-8"))
    del value["sources"][0]["account"]
    _write_json(case["source_spec"], value)
    invoked = False

    def should_not_run(_arguments):
        nonlocal invoked
        invoked = True
        return 1

    monkeypatch.setattr(consensus, "_run_snapshot_cli", should_not_run)

    with pytest.raises(consensus.ConsensusError, match="declaration bundle"):
        _prepare(case)

    assert invoked is False
    assert not case["attempt_a"].exists()
    assert not case["receipt"].exists()


def test_verified_manifest_disagreement_has_no_retry_and_installs_block_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    original = consensus._run_attempt
    calls = 0

    def divergent_attempt(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            case["rdf"].write_bytes(
                _public_rdf().replace(b"Public child", b"Changed child")
            )
        return result

    monkeypatch.setattr(consensus, "_run_attempt", divergent_attempt)

    receipt, exit_code = _prepare(case)

    assert calls == 2
    assert exit_code == 2
    assert receipt["accepted"] is False
    assert receipt["repeatability_verified"] is False
    assert receipt["graph_gate_pass"] is False
    assert receipt["reason"] == "exact_consensus_mismatch"
    assert receipt["common_snapshot_fingerprint"] is None
    assert receipt["common_records"] is None
    assert receipt["readiness"] is None
    assert receipt["comparison_checks"]["source_records_equal"] is False
    assert receipt["comparison_checks"]["full_manifest_equal"] is False
    assert case["receipt"].is_dir()


def test_legacy_only_disagreement_is_recorded_but_does_not_block_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path, with_legacy=True)
    original = consensus._run_attempt
    calls = 0

    def legacy_divergence(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            case["legacy"].write_bytes(b"1\t2\n")
        return result

    monkeypatch.setattr(consensus, "_run_attempt", legacy_divergence)

    receipt, exit_code = _prepare(case)

    assert calls == 2
    assert exit_code == 0
    assert receipt["accepted"] is True
    assert receipt["repeatability_verified"] is True
    assert receipt["graph_gate_pass"] is True
    assert receipt["reason"] == "exact_consensus_ready"
    assert receipt["comparison_checks"]["nonlegacy_manifest_equal"] is True
    assert receipt["comparison_checks"]["full_manifest_equal"] is False
    assert receipt["comparison_checks"]["legacy_parity_artifact_record_equal"] is False
    assert receipt["warnings"] == ["legacy_parity_disagreement"]


@pytest.mark.parametrize("bad_output", ("alias", "existing", "symlink"))
def test_nonfresh_or_aliased_outputs_fail_before_any_compile(
    tmp_path: Path, bad_output: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    if bad_output == "alias":
        case["attempt_b"] = case["attempt_a"]
    elif bad_output == "existing":
        case["attempt_a"].mkdir()
    else:
        target = case["local_root"] / "target"
        target.mkdir()
        case["attempt_a"].symlink_to(target, target_is_directory=True)
    invoked = False

    def should_not_run(_arguments):
        nonlocal invoked
        invoked = True
        return 1

    monkeypatch.setattr(consensus, "_run_snapshot_cli", should_not_run)

    with pytest.raises(consensus.ConsensusError):
        _prepare(case)

    assert invoked is False
    assert not case["receipt"].exists()


def test_symlinked_ancestor_is_rejected_before_any_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _case(real)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    case = {
        "attempt_a": linked / "local" / "attempt-a",
        "attempt_b": linked / "local" / "attempt-b",
        "local_root": linked / "local",
        "policy": linked / "inputs" / "policy.json",
        "receipt": linked / "local" / "consensus",
        "source_spec": linked
        / "local"
        / "declaration"
        / declaration.SPEC_FILENAME,
    }
    invoked = False

    def should_not_run(_arguments):
        nonlocal invoked
        invoked = True
        return 1

    monkeypatch.setattr(consensus, "_run_snapshot_cli", should_not_run)

    with pytest.raises(consensus.ConsensusError, match="declaration bundle"):
        _prepare(case)

    assert invoked is False


def test_empty_git_named_ancestor_is_not_a_worktree(tmp_path: Path) -> None:
    neutral = tmp_path / "neutral"
    neutral.mkdir()
    (neutral / ".git").mkdir()
    case = _case(neutral)

    receipt, exit_code = _prepare(case)

    assert exit_code == 0
    assert receipt["graph_gate_pass"] is True


def test_real_git_marker_ancestor_fails_before_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    case = _case(worktree)
    marker = worktree / ".git"
    marker.mkdir()
    (marker / "HEAD").write_text("ref: refs/heads/test\n", encoding="utf-8")
    invoked = False

    def should_not_run(_arguments):
        nonlocal invoked
        invoked = True
        return 1

    monkeypatch.setattr(consensus, "_run_snapshot_cli", should_not_run)

    with pytest.raises(consensus.ConsensusError, match="declaration bundle"):
        _prepare(case)

    assert invoked is False


@pytest.mark.parametrize("mutated_input", ("policy", "rdf-account"))
def test_source_spec_or_policy_mutation_is_an_operational_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated_input: str
) -> None:
    case = _case(tmp_path)

    def mutate_before_return(_arguments):
        if mutated_input == "policy":
            case["policy"].write_text("{}\n", encoding="utf-8")
        else:
            source_spec = json.loads(case["source_spec"].read_text(encoding="utf-8"))
            source_spec["sources"][0]["account"] = "different-explicit-account"
            _write_json(case["source_spec"], source_spec)
        return 1

    monkeypatch.setattr(consensus, "_run_snapshot_cli", mutate_before_return)

    with pytest.raises(consensus.ConsensusError, match="changed across attempts"):
        _prepare(case)

    assert not case["receipt"].exists()
    assert not case["attempt_b"].exists()


def test_operational_subprocess_failure_does_not_create_scientific_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    monkeypatch.setattr(consensus, "_run_snapshot_cli", lambda _arguments: 1)

    with pytest.raises(consensus.ConsensusError, match="operational failure"):
        _prepare(case)

    assert not case["receipt"].exists()


def test_output_parent_identity_is_rechecked_before_receipt_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    original_check = consensus._assert_directory_identities
    checks = 0
    installed = False

    def fail_before_receipt(bindings):
        nonlocal checks
        checks += 1
        if checks == 7:
            raise consensus.ConsensusError("simulated output parent replacement")
        original_check(bindings)

    def should_not_install(_receipt_dir, _receipt):
        nonlocal installed
        installed = True

    monkeypatch.setattr(consensus, "_assert_directory_identities", fail_before_receipt)
    monkeypatch.setattr(consensus, "_install_receipt", should_not_install)

    with pytest.raises(consensus.ConsensusError, match="parent replacement"):
        _prepare(case)

    assert checks == 7
    assert installed is False
    assert case["attempt_a"].is_dir() and case["attempt_b"].is_dir()
    assert not case["receipt"].exists()


def test_cli_stdout_is_aggregate_only_and_verify_preserves_scientific_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _case(tmp_path)
    arguments = [
        "prepare",
        "--source-spec",
        str(case["source_spec"]),
        "--relation-policy",
        str(case["policy"]),
        "--attempt-a-dir",
        str(case["attempt_a"]),
        "--attempt-b-dir",
        str(case["attempt_b"]),
        "--receipt-dir",
        str(case["receipt"]),
        "--local-root",
        str(case["local_root"]),
        "--local-only",
        "--minimum-anchors",
        "3",
        "--resource-ceiling-bytes",
        str(RESOURCE_CEILING),
    ]

    assert consensus.main(arguments) == 2
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output == {
        "accepted": False,
        "graph_asset_ready": False,
        "reason": "graph_asset_not_ready",
    }
    assert captured.err == ""
    assert str(tmp_path) not in captured.out
    assert "Public root" not in captured.out
    assert consensus.main(
        [
            "verify",
            "--receipt-dir",
            str(case["receipt"]),
            "--attempt-a-dir",
            str(case["attempt_a"]),
            "--attempt-b-dir",
            str(case["attempt_b"]),
            "--source-spec",
            str(case["source_spec"]),
            "--relation-policy",
            str(case["policy"]),
        ]
    ) == 2
    verified_output = json.loads(capsys.readouterr().out)
    assert verified_output == {
        "accepted": False,
        "reason": "graph_asset_not_ready",
        "verified": True,
    }


def test_receipt_verification_rejects_extra_file_and_wrong_modes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _prepare(case)
    extra = case["receipt"] / "extra"
    extra.write_text("not allowed", encoding="utf-8")

    with pytest.raises(consensus.ConsensusError, match="file set mismatch"):
        _verify(case)

    extra.unlink()
    os.chmod(case["receipt"] / consensus.RECEIPT_NAME, 0o644)
    with pytest.raises(consensus.ConsensusError, match="mode is not 0600"):
        _verify(case)


def test_verifier_rejects_copied_attempt_inside_git_worktree(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _prepare(case)
    worktree = tmp_path / "copied-worktree"
    worktree.mkdir()
    marker = worktree / ".git"
    marker.mkdir()
    (marker / "HEAD").write_text("ref: refs/heads/test\n", encoding="utf-8")
    copied_attempt = worktree / "attempt-a"
    shutil.copytree(case["attempt_a"], copied_attempt)

    with pytest.raises(consensus.ConsensusError, match="snapshot attempt.*Git worktree"):
        consensus.verify_consensus_receipt(
            case["receipt"],
            attempt_a_dir=copied_attempt,
            attempt_b_dir=case["attempt_b"],
            source_spec=case["source_spec"],
            relation_policy=case["policy"],
        )


def test_receipt_verification_rederives_decision_after_valid_reseal(tmp_path: Path) -> None:
    case = _case(tmp_path)
    receipt, _exit_code = _prepare(case)
    tampered = dict(receipt)
    del tampered["repeatability_contract_sha256"]
    tampered["accepted"] = False
    tampered["reason"] = "exact_consensus_mismatch"
    tampered = consensus._seal_receipt(tampered)
    (case["receipt"] / consensus.RECEIPT_NAME).write_bytes(
        consensus._canonical_json(tampered)
    )

    with pytest.raises(consensus.ConsensusError, match="decision does not follow"):
        _verify(case)


def test_receipt_verification_rejects_malformed_common_provenance_after_reseal(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    receipt, _exit_code = _prepare(case)
    tampered = copy.deepcopy(receipt)
    del tampered["repeatability_contract_sha256"]
    tampered["common_records"]["repository_commit"] = "0" * 39
    tampered = consensus._seal_receipt(tampered)
    (case["receipt"] / consensus.RECEIPT_NAME).write_bytes(
        consensus._canonical_json(tampered)
    )

    with pytest.raises(consensus.ConsensusError, match="common repeatability records"):
        _verify(case)


@pytest.mark.parametrize("target", ("privacy-rule", "aggregate-summary"))
def test_receipt_verification_rejects_coherent_resealed_forgery_against_attempts(
    tmp_path: Path, target: str
) -> None:
    case = _case(tmp_path)
    receipt, _exit_code = _prepare(case)
    tampered = copy.deepcopy(receipt)
    del tampered["repeatability_contract_sha256"]
    if target == "privacy-rule":
        tampered["common_records"]["compiler_implementation_records"][
            "privacy_rule"
        ] = {"sha256": "0" * 64, "size_bytes": 1}
    else:
        tampered["aggregate_summary"]["eligible_anchor_count"] += 10
    tampered = consensus._seal_receipt(tampered)
    (case["receipt"] / consensus.RECEIPT_NAME).write_bytes(
        consensus._canonical_json(tampered)
    )

    with pytest.raises(consensus.ConsensusError):
        _verify(case)


@pytest.mark.parametrize("substitution", ("source-spec", "relation-policy"))
def test_verifier_binds_supplied_inputs_to_both_attempt_manifests(
    tmp_path: Path, substitution: str
) -> None:
    case = _case(tmp_path)
    receipt, _exit_code = _prepare(case)
    tampered = copy.deepcopy(receipt)
    del tampered["repeatability_contract_sha256"]
    if substitution == "source-spec":
        alternate_dir = case["local_root"] / "alternate-declaration"
        alternate = declaration.build_source_spec(
            snapshot_label="different-valid-label",
            rdf=(("explicit-rdf", "groups", str(case["rdf"])),),
        )
        declaration.install_source_spec(
            alternate,
            output_dir=alternate_dir,
            local_root=case["local_root"],
        )
        supplied = alternate_dir / declaration.SPEC_FILENAME
        tampered["input_records"]["source_spec"] = consensus._regular_file_record(
            supplied, "alternate source spec"
        )
        source_spec = supplied
        relation_policy = case["policy"]
    else:
        supplied = case["policy"].parent / "alternate-policy.json"
        alternate_policy = {
            "physical_edges": {**PHYSICAL_RELATIONS, "cross_link": True},
            "schema": "pearltrees-diffusion-relation-policy-v1",
        }
        _write_json(supplied, alternate_policy)
        tampered["input_records"]["relation_policy"] = consensus._regular_file_record(
            supplied, "alternate relation policy"
        )
        source_spec = case["source_spec"]
        relation_policy = supplied
    tampered = consensus._seal_receipt(tampered)
    (case["receipt"] / consensus.RECEIPT_NAME).write_bytes(
        consensus._canonical_json(tampered)
    )

    with pytest.raises(consensus.ConsensusError, match="bind the supplied input records"):
        consensus.verify_consensus_receipt(
            case["receipt"],
            attempt_a_dir=case["attempt_a"],
            attempt_b_dir=case["attempt_b"],
            source_spec=source_spec,
            relation_policy=relation_policy,
        )


def test_verifier_rejects_one_attempt_directory_supplied_twice(tmp_path: Path) -> None:
    case = _case(tmp_path)
    _prepare(case)

    with pytest.raises(consensus.ConsensusError, match="two distinct nonnested"):
        consensus.verify_consensus_receipt(
            case["receipt"],
            attempt_a_dir=case["attempt_a"],
            attempt_b_dir=case["attempt_a"],
            source_spec=case["source_spec"],
            relation_policy=case["policy"],
        )


def test_verifier_detects_attempt_leaf_replacement_after_child_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    _prepare(case)
    original_run = consensus._run_snapshot_cli
    calls = 0

    def replace_after_verification(arguments):
        nonlocal calls
        result = original_run(arguments)
        calls += 1
        if calls == 1:
            case["attempt_a"].rename(case["local_root"] / "verified-attempt-a")
            shutil.copytree(case["attempt_b"], case["attempt_a"])
        return result

    monkeypatch.setattr(consensus, "_run_snapshot_cli", replace_after_verification)

    with pytest.raises(consensus.ConsensusError, match="identity changed"):
        _verify(case)


def test_compiler_launch_is_fixed_sibling_path_and_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = None

    class Completed:
        returncode = 1

    def fake_run(command, **kwargs):
        nonlocal observed
        observed = (command, kwargs)
        return Completed()

    monkeypatch.setattr(consensus.subprocess, "run", fake_run)

    assert consensus._run_snapshot_cli(("verify", "--run-dir", "/not-used")) == 1
    command, kwargs = observed
    assert command[:4] == [
        consensus.sys.executable,
        "-I",
        "-B",
        str(consensus.SNAPSHOT_PREPARER),
    ]
    assert kwargs["cwd"] == consensus.REPO_ROOT
    assert kwargs["capture_output"] is True

#!/usr/bin/env python3
"""Validate the frozen process-expression P1 preregistration contract."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from process_cards import REGISTRY_VERSION, RENDERER_VERSION, canonical, parse
from routed_policy import canonical_json_bytes, strict_json_loads


ROOT = Path(__file__).resolve().parent
PREREG_PATH = ROOT / "PROCESS_EXPRESSION_P1_PREREG.json"
SCHEMA = "unifyweaver.process-expression-p1-prereg.v1"


class ProtocolError(ValueError):
    """The process-expression P1 contract is malformed or has drifted."""


def _require(condition: bool, message: str):
    if not condition:
        raise ProtocolError(message)


def _full_process_digest(expression: str) -> str:
    node = parse(expression)
    return hashlib.sha256(
        f"{REGISTRY_VERSION}|{canonical(node)}".encode("utf-8")
    ).hexdigest()


def _prereg_id(document: Mapping[str, Any]) -> str:
    core = dict(document)
    core.pop("prereg_id", None)
    return hashlib.sha256(canonical_json_bytes(core)).hexdigest()


def load_and_verify(path: str | Path = PREREG_PATH) -> dict[str, Any]:
    path = Path(path)
    try:
        document = strict_json_loads(path.read_bytes(), source=str(path))
    except (OSError, ValueError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"cannot load P1 preregistration: {exc}") from exc
    _require(isinstance(document, dict), "preregistration must be a JSON object")
    _require(document.get("schema") == SCHEMA, "unsupported preregistration schema")

    protocol_path = path.parent / document.get("protocol_path", "")
    _require(protocol_path.is_file(), "protocol document is missing")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    _require(
        document.get("protocol_sha256") == protocol_sha,
        "protocol document hash differs from preregistration",
    )

    source = document.get("required_source_contract", {})
    _require(source.get("task_schema") == "unifyweaver.routed-task.v2", "task v2 required")
    _require(source.get("picks_schema") == "unifyweaver.routed-picks.v2", "picks v2 required")
    _require(
        source.get("execution_bundle_schema")
        == "unifyweaver.routed-execution-bundle.v1",
        "execution bundle v1 required",
    )
    _require(
        source.get("privacy_policy_id") == "pearltrees-public-only-v1",
        "public-only privacy policy required",
    )
    _require(
        source.get("catalog_policy_id")
        == "pearltrees-public-alphanumeric-title-v1",
        "public catalog policy required",
    )
    _require(
        source.get("missing_bundle_outcome")
        == "blocked_no_eligible_v2_labels",
        "missing eligible labels must block",
    )
    _require(
        document.get("legacy_sources", {}).get("legacy_unbound_picks_authorized") is False,
        "legacy picks must remain unauthorized",
    )
    _require(
        document.get("legacy_sources", {}).get("private_inclusive_sources_authorized") is False,
        "private-inclusive sources must remain unauthorized",
    )

    identity = document.get("process_identity", {})
    _require(identity.get("digest_hex_length") == 64, "full process SHA-256 required")
    _require(identity.get("compact_ast_sha_is_identity") is False, "compact AST hash is not identity")
    _require(
        identity.get("registry_version") == REGISTRY_VERSION,
        "process registry version drifted",
    )
    _require(
        identity.get("renderer_version") == RENDERER_VERSION,
        "process renderer version drifted",
    )
    processes = identity.get("processes")
    _require(isinstance(processes, dict) and len(processes) == 4, "four P1 processes required")
    process_digests = {}
    process_canonicals = {}
    for name, record in sorted(processes.items()):
        _require(isinstance(name, str) and isinstance(record, dict), "invalid process entry")
        expression = record.get("expression")
        _require(isinstance(expression, str), "process expression is missing")
        process_canonicals[name] = canonical(parse(expression))
        _require(
            record.get("canonical") == process_canonicals[name],
            f"canonical process drifted for {name}",
        )
        process_digests[name] = _full_process_digest(expression)
        _require(len(process_digests[name]) == 64, "process digest is truncated")
        _require(
            record.get("sha256") == process_digests[name],
            f"full process digest drifted for {name}",
        )
    _require(len(set(process_digests.values())) == len(process_digests), "process identities collide")

    ledger = document.get("ledger", {})
    _require(
        ledger.get("recorded_destination_in_training_ledger") is False,
        "training ledger must exclude recorded destinations",
    )
    _require(
        ledger.get("all_process_rows_for_query_share_split") is True,
        "process-complete query grouping required",
    )
    _require(ledger.get("null_primary_ranking_loss") == "excluded", "null loss rule drifted")
    _require(ledger.get("process_total_training_mass") == "equal", "process weighting drifted")

    split = document.get("split", {})
    _require(split.get("function") == "node_disjoint_pair_split", "wrong split function")
    _require(split.get("outer_seed") == 3980001, "outer seed drifted")
    _require(split.get("held_node_fraction") == 0.4, "held fraction drifted")
    _require(split.get("candidate_assignments") == 64, "split search drifted")
    _require(split.get("cross_rows") == "excluded", "cross rows must be excluded")

    training = document.get("training", {})
    _require(training.get("seeds") == [3980101, 3980102, 3980103], "training seeds drifted")
    _require(
        training.get("arms") == ["expression", "flat", "merged", "shuffled"],
        "training arms drifted",
    )
    _require(training.get("matched_budget_across_arms") is True, "arm budgets must match")
    _require(
        training.get("training_plan_required_before_fit") is True,
        "training plan must precede fitting",
    )
    _require(training.get("shuffled_train_permutations") == 5, "shuffle count drifted")
    probs = training.get("verbosity_probabilities", {})
    _require(set(probs) == {"V0", "V1", "V2", "V3"}, "verbosity levels drifted")
    _require(abs(sum(float(value) for value in probs.values()) - 1.0) < 1e-12, "verbosity mass != 1")
    _require(
        probs == {"V0": 0.1, "V1": 0.6, "V2": 0.25, "V3": 0.05},
        "verbosity probabilities drifted",
    )

    primary = document.get("primary", {})
    _require(primary.get("metric") == "exact-destination-mrr", "primary metric drifted")
    _require(primary.get("contrast") == "expression-minus-flat", "primary contrast drifted")
    _require(
        primary.get("superiority", {}).get("minimum_point_gain") == 0.01,
        "practical floor drifted",
    )
    _require(
        primary.get("superiority", {}).get("interval_lower_strictly_greater_than")
        == 0.0,
        "superiority interval gate drifted",
    )
    _require(
        primary.get("noninferiority_classification_only", {}).get("interval_lower_at_least")
        == -0.005,
        "noninferiority margin drifted",
    )
    _require(
        primary.get("noninferiority_classification_only", {}).get("counts_as_superiority")
        is False,
        "noninferiority must not count as superiority",
    )
    _require(primary.get("e5_deployment_safety_margin") == -0.01, "e5 safety margin drifted")
    bootstrap = primary.get("bootstrap", {})
    _require(bootstrap.get("function") == "paired_node_bootstrap_ci", "bootstrap drifted")
    _require(bootstrap.get("resamples") == 9999, "bootstrap count drifted")
    _require(bootstrap.get("seed") == 3980999, "bootstrap seed drifted")
    _require(bootstrap.get("confidence") == 0.95, "bootstrap confidence drifted")
    _require(
        bootstrap.get("minimum_endpoint_components_for_decision") == 20,
        "bootstrap component floor drifted",
    )

    policy = document.get("primary_policy_process", {})
    _require(policy.get("margin_lt_0.02") == "sonnet_lineage_n10", "low-band policy drifted")
    _require(
        policy.get("margin_ge_0.02_lt_0.03") == "sonnet_lineage_n20",
        "middle-band policy drifted",
    )
    _require(policy.get("margin_ge_0.03") == "e5_auto", "high-band policy drifted")

    multiplicity = document.get("multiplicity", {})
    _require(
        multiplicity.get("expression_vs_flat_is_only_decision_bearing_contrast") is True,
        "primary multiplicity rule drifted",
    )
    _require(
        multiplicity.get("secondary_results_are_descriptive") is True,
        "secondary analyses must remain descriptive",
    )
    _require(
        multiplicity.get("p3_cannot_retroactively_change_p1") is True,
        "P3 must not retroactively change P1",
    )

    _require(document.get("prereg_id") == _prereg_id(document), "preregistration ID mismatch")
    document["_derived_process_sha256"] = process_digests
    document["_derived_process_canonical"] = process_canonicals
    return document


def main():
    document = load_and_verify()
    print(document["prereg_id"])
    for name, digest in document["_derived_process_sha256"].items():
        print(f"{name}\t{digest}")


if __name__ == "__main__":
    main()

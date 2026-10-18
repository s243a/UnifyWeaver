"""Enforcement tests for the prospective process-expression P1 contract."""

import hashlib
import json
from pathlib import Path

import pytest

from process_cards import ast_sha, parse
from process_expression_p1_protocol import (
    PREREG_PATH,
    ProtocolError,
    _full_process_digest,
    _prereg_id,
    load_and_verify,
)


def _document():
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _write_fixture(tmp_path, document, protocol_bytes=None):
    source_protocol = PREREG_PATH.with_name(document["protocol_path"])
    protocol_bytes = protocol_bytes if protocol_bytes is not None else source_protocol.read_bytes()
    protocol = tmp_path / document["protocol_path"]
    protocol.write_bytes(protocol_bytes)
    document["protocol_sha256"] = hashlib.sha256(protocol_bytes).hexdigest()
    document["prereg_id"] = _prereg_id(document)
    path = tmp_path / PREREG_PATH.name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_committed_preregistration_verifies():
    document = load_and_verify()
    assert document["primary"]["contrast"] == "expression-minus-flat"
    assert len(document["_derived_process_sha256"]) == 4


def test_process_identity_is_full_registry_bound_sha():
    document = load_and_verify()
    # The preregistration is sealed under the registry version it records
    # (v0.3); digests recompute under that recorded version, while parsing
    # and canonicalization run under the current registry.
    recorded = document["process_identity"]["registry_version"]
    for name, record in document["process_identity"]["processes"].items():
        expression = record["expression"]
        full = _full_process_digest(expression, recorded)
        assert len(full) == 64
        assert record["sha256"] == full
        assert record["canonical"] == document["_derived_process_canonical"][name]
        assert document["_derived_process_sha256"][name] == full


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda d: d["legacy_sources"].update(legacy_unbound_picks_authorized=True),
            "legacy picks",
        ),
        (
            lambda d: d["primary"].update(contrast="expression-minus-merged"),
            "primary contrast",
        ),
        (
            lambda d: d["split"].update(outer_seed=1),
            "outer seed",
        ),
        (
            lambda d: d["training"].update(seeds=[1, 2, 3]),
            "training seeds",
        ),
        (
            lambda d: d["process_identity"].update(registry_version="v0.0"),
            "registry version",
        ),
        (
            lambda d: d["process_identity"]["processes"]["e5_auto"].update(
                sha256="0" * 64
            ),
            "full process digest",
        ),
        (
            lambda d: d["required_source_contract"].update(
                catalog_policy_id="private-inclusive"
            ),
            "public catalog policy",
        ),
        (
            lambda d: d["primary"]["noninferiority_classification_only"].update(
                counts_as_superiority=True
            ),
            "must not count as superiority",
        ),
    ],
)
def test_resealed_material_mutations_still_fail(tmp_path, mutation, message):
    document = _document()
    mutation(document)
    path = _write_fixture(tmp_path, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify(path)


def test_protocol_text_tamper_fails_without_resealing(tmp_path):
    document = _document()
    path = _write_fixture(tmp_path, document)
    protocol = path.with_name(document["protocol_path"])
    protocol.write_bytes(protocol.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="protocol document hash"):
        load_and_verify(path)


def test_preregistration_id_tamper_fails(tmp_path):
    document = _document()
    path = _write_fixture(tmp_path, document)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["prereg_id"] = "0" * 64
    path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="ID mismatch"):
        load_and_verify(path)

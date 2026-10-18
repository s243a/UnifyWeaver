"""Enforcement tests for the frozen SM-FS ranking and transfer contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sm_fs_protocols import (
    ProtocolError,
    RANKING_PREREG_PATH,
    TRANSFER_PREREG_PATH,
    _prereg_id,
    load_and_verify_ranking,
    load_and_verify_transfer,
    sampler_index,
    sampler_key_bytes,
)


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path, source: Path, document: dict) -> Path:
    protocol_source = source.with_name(document["protocol_path"])
    protocol = tmp_path / document["protocol_path"]
    protocol.write_bytes(protocol_source.read_bytes())
    document["protocol_sha256"] = hashlib.sha256(protocol.read_bytes()).hexdigest()
    document["prereg_id"] = _prereg_id(document)
    path = tmp_path / source.name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_committed_protocols_verify_and_transfer_remains_blocked():
    ranking = load_and_verify_ranking()
    transfer = load_and_verify_transfer()
    assert ranking["constructor_implementation_authorized"] is True
    assert ranking["model_fitting_authorized"] is False
    assert ranking["reserve"]["scored_by_protocol"] is False
    assert transfer["execution_authorized"] is False
    assert transfer["source"]["negative_bundle_sha256"] is None


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["source_bundle"].update(ledger_catalog_authorized=True),
            "ledger catalog",
        ),
        (
            lambda d: d["source_bundle"].update(reserve_rows_authorized=True),
            "reserve rows",
        ),
        (
            lambda d: d["construction"]["counts"].update(nonancestors=126806),
            "construction counts",
        ),
        (
            lambda d: d["construction"]["query_mass"].update(positive=0.6),
            "positive query mass",
        ),
        (
            lambda d: d["construction"]["graph_target"].update(decay=0.6),
            "graph-target decay",
        ),
        (
            lambda d: d["construction"]["hardness"].update(
                hard="reachable-distance-1-through-3"
            ),
            "hardness definitions",
        ),
        (
            lambda d: d["split"].update(assignment_sha256="0" * 64),
            "fold assignment hash",
        ),
        (
            lambda d: d["training"].update(seeds=[1, 2, 3]),
            "ranking seeds",
        ),
        (
            lambda d: d["training"]["sampler"].update(
                kind="uniform-row-sampling"
            ),
            "minibatch sampler",
        ),
        (
            lambda d: d["training"].update(path_components_in_embedding_text=True),
            "path components",
        ),
        (
            lambda d: d["primary"].update(contrast="graded-negative-minus-warm-start"),
            "primary contrast",
        ),
        (
            lambda d: d["primary"]["bootstrap"].update(unit="row"),
            "bootstrap unit",
        ),
        (
            lambda d: d["primary"].update(
                interval_lower_strictly_greater_than=False
            ),
            "interval gate",
        ),
        (
            lambda d: d["reserve"].update(scored_by_protocol=True),
            "reserve must remain unscored",
        ),
        (
            lambda d: d["privacy"].update(checkpoint_release_authorized=True),
            "checkpoint release",
        ),
        (
            lambda d: d.update(model_fitting_authorized=True),
            "model fitting must remain blocked",
        ),
    ],
)
def test_resealed_ranking_mutations_fail(tmp_path, mutation, message):
    document = _document(RANKING_PREREG_PATH)
    mutation(document)
    path = _write_fixture(tmp_path, RANKING_PREREG_PATH, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_ranking(path)


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda d: d["source"].update(negative_bundle_sha256="0" * 64),
            "unbound negative bundle",
        ),
        (
            lambda d: d["source"].update(ranking_prereg_id="0" * 64),
            "source ranking preregistration",
        ),
        (
            lambda d: d["training"].update(pearltrees_outcome_selection_authorized=True),
            "target outcomes",
        ),
        (
            lambda d: d["null_control"].update(required=False),
            "matched source control",
        ),
        (
            lambda d: d["transfer"].update(specificity_contrast="T1-minus-T0"),
            "specificity contrast",
        ),
        (
            lambda d: d["transfer"].update(
                total_interval_lower_strictly_greater_than=-99
            ),
            "total-transfer interval gate",
        ),
        (
            lambda d: d["transfer"].update(correct_candidate_lineage_primary=True),
            "candidate-lineage",
        ),
        (
            lambda d: d["target"].update(catalog_frozen_before_placement_labels=False),
            "catalog timing",
        ),
        (
            lambda d: d["target"].update(
                e5_revision="0000000000000000000000000000000000000000"
            ),
            "target E5 revision",
        ),
        (
            lambda d: d.update(execution_authorized=True),
            "must not authorize execution",
        ),
        (
            lambda d: d["privacy"].update(checkpoint_release_authorized=True),
            "checkpoint release",
        ),
        (
            lambda d: d["privacy"].update(
                provider_calls_with_source_content=True
            ),
            "source content",
        ),
    ],
)
def test_resealed_transfer_mutations_fail(tmp_path, mutation, message):
    document = _document(TRANSFER_PREREG_PATH)
    mutation(document)
    path = _write_fixture(tmp_path, TRANSFER_PREREG_PATH, document)
    with pytest.raises(ProtocolError, match=message):
        load_and_verify_transfer(path)


@pytest.mark.parametrize(
    "source,loader",
    [
        (RANKING_PREREG_PATH, load_and_verify_ranking),
        (TRANSFER_PREREG_PATH, load_and_verify_transfer),
    ],
)
def test_protocol_text_tamper_fails_without_resealing(tmp_path, source, loader):
    document = _document(source)
    path = _write_fixture(tmp_path, source, document)
    protocol = path.with_name(document["protocol_path"])
    protocol.write_bytes(protocol.read_bytes() + b"\nchanged\n")
    with pytest.raises(ProtocolError, match="protocol document hash"):
        loader(path)


@pytest.mark.parametrize(
    "source,loader",
    [
        (RANKING_PREREG_PATH, load_and_verify_ranking),
        (TRANSFER_PREREG_PATH, load_and_verify_transfer),
    ],
)
def test_preregistration_id_tamper_fails(tmp_path, source, loader):
    document = _document(source)
    path = _write_fixture(tmp_path, source, document)
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["prereg_id"] = "0" * 64
    path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="ID mismatch"):
        loader(path)


def test_protocol_path_must_be_the_exact_sibling(tmp_path):
    document = _document(RANKING_PREREG_PATH)
    path = _write_fixture(tmp_path, RANKING_PREREG_PATH, document)
    outside = tmp_path.parent / "outside.md"
    outside.write_bytes(
        path.with_name(document["protocol_path"]).read_bytes()
    )
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["protocol_path"] = "../outside.md"
    changed["protocol_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    changed["prereg_id"] = _prereg_id(changed)
    path.write_text(json.dumps(changed, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="protocol path"):
        load_and_verify_ranking(path)


def test_protocol_symlink_is_rejected(tmp_path):
    document = _document(RANKING_PREREG_PATH)
    real_protocol = tmp_path / "real.md"
    real_protocol.write_bytes(
        RANKING_PREREG_PATH.with_name(document["protocol_path"]).read_bytes()
    )
    protocol = tmp_path / document["protocol_path"]
    protocol.symlink_to(real_protocol)
    document["protocol_sha256"] = hashlib.sha256(real_protocol.read_bytes()).hexdigest()
    document["prereg_id"] = _prereg_id(document)
    path = tmp_path / RANKING_PREREG_PATH.name
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="symlink"):
        load_and_verify_ranking(path)


def test_hard_bound_id_rejects_resealed_unenumerated_material_change(tmp_path):
    document = _document(RANKING_PREREG_PATH)
    document["prior_pilot"]["reproduced_tuned_row_correlation"] = 0.81
    path = _write_fixture(tmp_path, RANKING_PREREG_PATH, document)
    with pytest.raises(ProtocolError, match="frozen ranking preregistration ID"):
        load_and_verify_ranking(path)


def test_sampler_known_answer_key_and_index():
    key = sampler_key_bytes(
        fold=0,
        seed=3997001,
        step=0,
        draw=0,
        role="query",
    )
    assert key == (
        b'{"bucket":"","draw":0,"fold":0,"query_id":"","retry":0,'
        b'"role":"query","sampler_id":"sm-fs-ranking-sampler-v1",'
        b'"schema":"unifyweaver.sm-fs-ranking-sampler-key.v1",'
        b'"seed":3997001,"step":0}\n'
    )
    assert sampler_index(
        361,
        fold=0,
        seed=3997001,
        step=0,
        draw=0,
        role="query",
    ) == (
        103,
        "12ccc6d6211021a2e2c210bee949e06048b08b573b80cd5274465d9acd489939",
        0,
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            dict(
                fold=0,
                seed=3997001,
                step=0,
                draw=0,
                role="common-positive",
            ),
            "must bind the selected query",
        ),
        (
            dict(
                fold=0,
                seed=3997001,
                step=0,
                draw=0,
                role="common-positive",
                query_id="Q",
                bucket="hard",
            ),
            "only a negative-candidate",
        ),
        (
            dict(
                fold=0,
                seed=3997001,
                step=0,
                draw=0,
                role="negative-candidate",
                query_id="Q",
            ),
            "must bind its bucket",
        ),
    ],
)
def test_sampler_role_domains_fail_closed(kwargs, message):
    with pytest.raises(ProtocolError, match=message):
        sampler_key_bytes(**kwargs)

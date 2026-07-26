#!/usr/bin/env python3
"""Contract-fixture tests: full process identity, resolved DTO, role paths.

Covers the step-1 slice of the minimum test list in
``DESIGN_expression_encoder_future.md`` §11: grammar-example round trips,
atom/operator dual roles, nested args/kwargs, lists, UTF-8/escaped strings,
pins, numeric boundary cases, and role-path non-aliasing — plus the identity
helpers that keep a compact ``ast_sha`` out of handoff artifacts.

The expected token stream for the worked example is written out by hand rather
than captured from the implementation, so the fixture constrains the code and
not the other way round.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
from process_cards import PROCESSES, REGISTRY, REGISTRY_VERSION
import process_expression_contract as pec
from process_expression_contract import (
    CONTRACT_VERSION,
    KIND_APPLY,
    KIND_ATOM,
    ContractError,
    build_golden,
    depth_breadth,
    load_golden,
    resolve,
    resolve_expression,
    role_paths,
    serialize_path,
    token_stream,
    verify_golden,
)
from process_identity import (
    ProcessIdentityError,
    deployed_identity,
    full_ast_digest,
    full_ast_digest_for_expression,
    promote_synthetic,
    require_full_digest,
    synthetic_identity,
    synthetic_sample_digest,
    verify_identity_record,
)

GOLDEN_PATH = ROOT / "PROCESS_EXPRESSION_GOLDEN_v2.json"
SUPERSEDED_GOLDEN_PATH = ROOT / "PROCESS_EXPRESSION_GOLDEN_v1.json"
SPEC_SHA = "a" * 64


def _stream(expression: str) -> list[str]:
    return [token.as_line() for token in token_stream(resolve_expression(expression))]


# --------------------------------------------------------------------------
# identity: the full digest, and refusing the compact one
# --------------------------------------------------------------------------


def test_full_digest_matches_the_frozen_p1_convention():
    """The handoff must not introduce a second digest convention (§2)."""

    from process_expression_p1_protocol import _full_process_digest

    for expression in PROCESSES.values():
        assert full_ast_digest_for_expression(expression) == _full_process_digest(
            expression
        )


def test_compact_ast_sha_is_a_prefix_but_not_an_identity():
    node = pc.parse("lineage(graph,decay=0.85)")
    full = full_ast_digest(node)
    compact = pc.ast_sha(node)
    assert len(full) == 64 and len(compact) == 16
    assert full.startswith(compact)  # P0 behavior is unchanged...
    with pytest.raises(ProcessIdentityError, match="compact 16-hex"):
        require_full_digest(compact)  # ...but it is not a handoff identity


def test_deployed_identity_binds_bytes_registry_digest_and_factory():
    node = pc.parse("kalman(luna.D,luna.S)")
    identity = deployed_identity(node, factory_fingerprint="factory-abc")
    assert identity.canonical_bytes == b"kalman(luna.D,luna.S)"
    assert identity.registry_version == REGISTRY_VERSION
    assert identity.full_digest == full_ast_digest(node)
    # Residual rows map by the full deployed identity, never the digest alone.
    assert identity.identity_key == f"{identity.full_digest}|factory-abc"
    verify_identity_record(identity.as_record())


def test_deployed_identity_requires_a_factory_fingerprint():
    node = pc.parse("kalman(luna.D,luna.S)")
    with pytest.raises(ProcessIdentityError, match="factory_fingerprint"):
        deployed_identity(node, factory_fingerprint="")


def test_identity_record_is_recomputed_from_its_own_bytes():
    node = pc.parse("lineage(graph,decay=0.85)")
    record = deployed_identity(node, factory_fingerprint="f").as_record()

    tampered = dict(record, full_process_digest="0" * 64)
    with pytest.raises(ProcessIdentityError, match="does not match its own bytes"):
        verify_identity_record(tampered)

    # A non-canonical spelling of the same process is rejected, not normalized.
    noncanonical = dict(record, canonical_identity_string="lineage(graph)")
    with pytest.raises(ProcessIdentityError, match="not canonical"):
        verify_identity_record(noncanonical)

    stale = dict(record, registry_version="v0.2")
    with pytest.raises(ProcessIdentityError, match="identity-version migration"):
        verify_identity_record(stale)


def test_synthetic_samples_bind_a_generator_spec_and_cannot_be_promoted():
    node = pc.parse("blend(luna.D,luna.S)")
    sample = synthetic_identity(node, generator_spec_sha256=SPEC_SHA)
    assert sample.synthetic_sample_digest == synthetic_sample_digest(
        full_ast_digest(node), SPEC_SHA
    )
    record = sample.as_record()
    assert record["synthetic_only"] is True
    verify_identity_record(record)

    with pytest.raises(ProcessIdentityError, match="cannot be promoted"):
        promote_synthetic(record, factory_fingerprint="factory-abc")

    forged = dict(record, factory_fingerprint="factory-abc")
    with pytest.raises(ProcessIdentityError, match="cannot carry a factory"):
        verify_identity_record(forged)

    drifted = dict(record, synthetic_sample_digest="0" * 64)
    with pytest.raises(ProcessIdentityError, match="does not bind this AST"):
        verify_identity_record(drifted)


def test_deployed_identity_is_not_a_generated_sample():
    node = pc.parse("blend(luna.D,luna.S)")
    record = deployed_identity(node, factory_fingerprint="f").as_record()
    with pytest.raises(ProcessIdentityError, match="not a generated sample"):
        verify_identity_record(dict(record, generator_spec_sha256=SPEC_SHA))


def test_distinct_registered_processes_have_distinct_identities():
    digests = {name: full_ast_digest_for_expression(e) for name, e in PROCESSES.items()}
    assert len(set(digests.values())) == len(digests)


# --------------------------------------------------------------------------
# resolved DTO: derived fields agree with the registry and the canonicalizer
# --------------------------------------------------------------------------


def test_resolved_kwargs_match_the_canonical_identity_string():
    """The DTO serializes exactly the kwargs canonical() renders (§3)."""

    for expression in list(PROCESSES.values()) + [
        "menu(graph,n=10)",
        "lineage(graph)",
        "lineage(graph,decay=0.85)",
    ]:
        node = pc.parse(expression)
        resolved = resolve(node)
        rendered = ",".join(
            f"{kw.key}={pc._render_val(kw.value)}" for kw in resolved.kwargs
        )
        # Every resolved kwarg appears verbatim in the identity string.
        for part in filter(None, rendered.split(",")):
            assert part in pc.canonical(node)


def test_elided_default_resolves_to_the_same_process():
    """An explicitly written default is the same process as its elided form."""

    bare = resolve_expression("lineage(graph)")
    explicit = resolve_expression("lineage(graph,decay=0.85)")
    assert bare == explicit
    assert full_ast_digest_for_expression("lineage(graph)") == (
        full_ast_digest_for_expression("lineage(graph,decay=0.85)")
    )


def test_atom_operator_dual_role_is_distinguished_by_kind():
    """`e5` is registered atom *and* operator; KIND separates the two uses."""

    assert REGISTRY["e5"].atom and REGISTRY["e5"].operator
    assert resolve_expression("e5").kind == KIND_ATOM
    assert resolve_expression("e5(margin(t=0.03))").kind == KIND_APPLY
    # KIND never overrides the registry: OUTPUT comes from the signature.
    assert resolve_expression("e5").output == REGISTRY["e5"].output


def test_output_type_comes_from_the_pinned_registry():
    for name, signature in REGISTRY.items():
        if not signature.atom or signature.operator:
            continue
        assert resolve_expression(name).output == signature.output


def test_variadic_positional_types_resolve():
    resolved = resolve_expression("blend(luna.D,luna.S,graph)")
    tokens = token_stream(resolved)
    paths = role_paths(tokens)
    assert "ARG(2,source)" in paths  # third arg uses the variadic type


def test_non_finite_and_unsupported_literals_fail_closed():
    from process_expression_contract import _value_type

    with pytest.raises(ContractError, match="booleans"):
        _value_type(True, None)
    with pytest.raises(ContractError, match="unsupported literal shape"):
        _value_type(object(), None)
    # Non-finite numerics are already rejected by the canonicalizer.
    with pytest.raises(ValueError):
        pc._render_val(float("inf"))


def test_token_stream_requires_a_resolved_dto():
    with pytest.raises(ContractError, match="resolved DTO"):
        token_stream(pc.parse("graph"))


# --------------------------------------------------------------------------
# the worked example, written out by hand
# --------------------------------------------------------------------------


EXPECTED_LINEAGE_STREAM = [
    "<BOS>\tROOT",
    "<NODE>\tROOT",
    "<KIND:apply>\tROOT",
    "<NAME:lineage>\tROOT",
    "<OUTPUT:target-set>\tROOT",
    "<ARGS>\tROOT",
    "<ARG:0>\tROOT",
    "<NODE>\tARG(0,source)",
    "<KIND:atom>\tARG(0,source)",
    "<NAME:graph>\tARG(0,source)",
    "<OUTPUT:source>\tARG(0,source)",
    "<ARGS>\tARG(0,source)",
    "</ARGS>\tARG(0,source)",
    "<KWARGS>\tARG(0,source)",
    "</KWARGS>\tARG(0,source)",
    "<MODS>\tARG(0,source)",
    "</MODS>\tARG(0,source)",
    "<PINS>\tARG(0,source)",
    "</PINS>\tARG(0,source)",
    "</NODE>\tARG(0,source)",
    "</ARG>\tROOT",
    "</ARGS>\tROOT",
    "<KWARGS>\tROOT",
    "<KW:decay>\tROOT",
    "<NUMBER>\tKWARG(decay,number)",
    "BYTE:0x30\tKWARG(decay,number)/LITERAL_BYTE(0)",
    "BYTE:0x2e\tKWARG(decay,number)/LITERAL_BYTE(1)",
    "BYTE:0x38\tKWARG(decay,number)/LITERAL_BYTE(2)",
    "BYTE:0x35\tKWARG(decay,number)/LITERAL_BYTE(3)",
    "</NUMBER>\tKWARG(decay,number)",
    "</KW>\tROOT",
    "</KWARGS>\tROOT",
    "<MODS>\tROOT",
    "</MODS>\tROOT",
    "<PINS>\tROOT",
    "</PINS>\tROOT",
    "</NODE>\tROOT",
    "<EOS>\tROOT",
]


def test_worked_example_matches_the_hand_written_stream():
    assert _stream("lineage(graph,decay=0.85)") == EXPECTED_LINEAGE_STREAM
    # "0.85" is the canonicalizer's own lexical form, byte for byte.
    assert bytes(
        int(line.split("\t")[0][7:], 16)
        for line in EXPECTED_LINEAGE_STREAM
        if line.startswith("BYTE:")
    ) == b"0.85"


def test_root_has_the_empty_path_and_root_is_not_an_edge_step():
    tokens = token_stream(resolve_expression("graph.discrim"))
    assert tokens[0].path == ()
    assert serialize_path(()) == "ROOT"
    # ROOT never appears as a role step inside a longer path.
    for token in tokens:
        assert "ROOT" not in serialize_path(token.path)[1:]


# --------------------------------------------------------------------------
# role paths: the complete path carries what (depth, breadth) loses
# --------------------------------------------------------------------------


def test_complete_paths_separate_positions_that_depth_breadth_aliases():
    """The concrete case §3.2 exists to prevent."""

    tokens = token_stream(
        resolve_expression("e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20]))")
    )
    groups: dict[tuple[int, int | None], set[str]] = {}
    for token in tokens:
        groups.setdefault(depth_breadth(token.path), set()).add(
            serialize_path(token.path)
        )
    aliased = {key: value for key, value in groups.items() if len(value) > 1}
    assert aliased, "expected the lossy coordinate to collide somewhere"

    # t[0] and menus[0] share (depth 3, index 0) but are different positions.
    assert {
        "ARG(0,process)/KWARG(t,number_list)/LIST_ITEM(0,number)",
        "ARG(0,process)/KWARG(menus,int_list)/LIST_ITEM(0,int)",
    } <= aliased[(3, 0)]


def test_sibling_swap_changes_the_path_and_the_stream():
    left = _stream("kalman(luna.D,luna.S)")
    right = _stream("kalman(luna.S,luna.D)")
    assert left != right
    assert "ARG(0,source)" in role_paths(
        token_stream(resolve_expression("kalman(luna.D,luna.S)"))
    )


def test_ordered_roles_are_not_commutative():
    """`(rho_a, rho_b)` must differ from `(rho_b, rho_a)` where both are valid."""

    paths = role_paths(
        token_stream(resolve_expression("distill(e5(routing(e5,haiku,t=[0.02],menus=[10])))"))
    )
    assert "ARG(0,score)/ARG(0,process)" in paths
    assert "ARG(0,process)/ARG(0,score)" not in paths


def test_keyword_source_order_canonicalizes_to_one_keyword_path():
    a = _stream('routing(e5,haiku,t=[0.02],menus=[10],manifest="m")')
    b = _stream('routing(e5,haiku,manifest="m",menus=[10],t=[0.02])')
    assert a == b  # canonicalization already sorts keywords


def test_kwarg_roles_use_the_registry_key_and_value_type():
    paths = role_paths(
        token_stream(resolve_expression("routing(e5,haiku,t=[0.02],menus=[10])"))
    )
    assert "KWARG(t,number_list)" in paths
    assert "KWARG(menus,int_list)" in paths
    assert "KWARG(t,number_list)/LIST_ITEM(0,number)" in paths
    assert "KWARG(menus,int_list)/LIST_ITEM(0,int)" in paths


def test_modifiers_and_pins_have_ordered_roles():
    paths = role_paths(
        token_stream(resolve_expression("lineage(graph,decay=0.85)@run/2026-07-25"))
    )
    assert any(path.startswith("PIN(0)") for path in paths)
    mods = role_paths(token_stream(resolve_expression("llm.element")))
    assert any(path.startswith("MOD(0)") for path in mods)


def test_registry_currently_caps_modifiers_at_one_per_node():
    """``MOD(index>0)`` is schema-legal but unreachable under registry v0.3.

    The role schema keeps ordered ``MOD(index)`` because the grammar could
    relax this later; recording the current cap here means the step-2 envelope
    scan measures the limit instead of assuming it.
    """

    def walk(node):
        yield node
        for child in node.args:
            yield from walk(child)

    with pytest.raises(pc.ParseError, match="at most one modifier"):
        pc.parse("llm.element.subcat")
    # Modifiers sit on nested atoms (`luna.D`), not on the root operator.
    observed = [
        len(n.mods) for e in PROCESSES.values() for n in walk(pc.parse(e))
    ]
    assert max(observed) == 1


def test_utf8_and_escaped_strings_keep_exact_bytes():
    tokens = token_stream(
        resolve_expression('routing(e5,haiku,t=[0.02],menus=[10],manifest="héllo·wörld")')
    )
    payload = bytes(
        int(token.token[7:], 16)
        for token in tokens
        if token.token.startswith("BYTE:")
        and "KWARG(manifest,string)" in serialize_path(token.path)
    )
    assert payload == "héllo·wörld".encode("utf-8")

    escaped = token_stream(
        resolve_expression('routing(e5,haiku,t=[0.02],menus=[10],manifest="a\\"b\\\\c")')
    )
    payload = bytes(
        int(token.token[7:], 16)
        for token in escaped
        if token.token.startswith("BYTE:")
        and "KWARG(manifest,string)" in serialize_path(token.path)
    )
    # The stream carries the decoded string, not its JSON quoting.
    assert payload == b'a"b\\c'


def test_every_registered_process_produces_a_well_formed_stream():
    for name, expression in PROCESSES.items():
        tokens = token_stream(resolve_expression(expression))
        assert tokens[0].token == "<BOS>" and tokens[-1].token == "<EOS>", name
        opens = sum(1 for t in tokens if t.token == "<NODE>")
        closes = sum(1 for t in tokens if t.token == "</NODE>")
        assert opens == closes, name


# --------------------------------------------------------------------------
# golden vectors
# --------------------------------------------------------------------------


def test_golden_fixture_verifies_and_covers_the_required_shapes():
    document = load_golden(GOLDEN_PATH)
    assert document["contract_version"] == CONTRACT_VERSION
    assert document["registry_version"] == REGISTRY_VERSION
    names = {row["name"] for row in document["rows"]}
    assert set(PROCESSES) <= names
    for required in (
        "int-spelled-number",
        "int-spelled-number-list",
        "atom-bare",
        "atom-dual-bare",
        "pinned",
        "utf8-string",
        "escaped-string",
        "menu-required-int",
        "blend-variadic",
        "neg-number",
    ):
        assert required in names, required


def test_golden_rows_carry_the_full_digest_not_the_compact_key():
    for row in load_golden(GOLDEN_PATH)["rows"]:
        require_full_digest(row["full_process_digest"])
        assert row["full_process_digest"] == full_ast_digest_for_expression(
            row["expression"]
        )


def test_golden_detects_content_drift(tmp_path):
    document = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    document["rows"][0]["tokens"][0] = "<NOPE>\tROOT"
    with pytest.raises(ContractError, match="golden_sha256"):
        verify_golden(document)


def test_golden_detects_a_row_recomputed_differently():
    document = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    document["rows"][0]["token_count"] = 9999
    core = {k: v for k, v in document.items() if k != "golden_sha256"}
    document["golden_sha256"] = pec.hashlib.sha256(
        pec._canonical_json_bytes(core)
    ).hexdigest()
    # Digest re-bound, so only recomputation from the expression catches it.
    with pytest.raises(ContractError, match="golden row drifted"):
        verify_golden(document)


def test_superseded_v1_bundle_is_retained_and_fails_closed():
    """A contract change creates a new bundle; it never mutates a sealed one.

    pec-v1 stays on disk as the sealed artifact it was.  Because scalar literals
    now carry their declared registry kind, it is loadable but no longer valid
    under the current contract — which is the designed behavior, not clutter.
    """

    assert SUPERSEDED_GOLDEN_PATH.exists()
    with pytest.raises(ContractError, match="different contract version"):
        load_golden(SUPERSEDED_GOLDEN_PATH)

    superseded = json.loads(SUPERSEDED_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert superseded["contract_version"] == "pec-v1"
    # No pec-v1 row was affected by the fix, which is exactly why pec-v2 had to
    # add explicit coverage for an integer-spelled `number`.
    assert not any(row["name"].startswith("int-spelled") for row in superseded["rows"])


def test_superseded_bundle_integrity_is_pinned_independently():
    """Version rejection happens before the checksum, so pin bytes separately.

    ``verify_golden`` raises on ``contract_version`` before it ever reaches
    ``golden_sha256``, so a *corrupted* pec-v1 raises the same error as an
    intact one.  Retaining a bundle as audit provenance is only meaningful if
    its bytes are actually pinned, hence this file-level hash.
    """

    import hashlib

    from process_expression_contract import SUPERSEDED_GOLDEN_BUNDLES

    for name, record in SUPERSEDED_GOLDEN_BUNDLES.items():
        path = ROOT / name
        assert path.exists(), name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"{name} has been mutated"
        assert json.loads(path.read_text(encoding="utf-8"))["contract_version"] == (
            record["contract_version"]
        )

    # Demonstrate the gap this closes: a corrupted v1 is indistinguishable from
    # an intact one through the loader alone.
    corrupted = json.loads(SUPERSEDED_GOLDEN_PATH.read_text(encoding="utf-8"))
    corrupted["rows"][0]["tokens"][0] = "<CORRUPTED>\tROOT"
    with pytest.raises(ContractError, match="different contract version"):
        verify_golden(corrupted)


def test_current_bundle_pointer_matches_the_loaded_fixture():
    from process_expression_contract import CURRENT_GOLDEN_BUNDLE

    assert GOLDEN_PATH.name == CURRENT_GOLDEN_BUNDLE
    assert SUPERSEDED_GOLDEN_PATH.name != CURRENT_GOLDEN_BUNDLE


def test_committed_bundle_is_reproducible_from_the_module_alone():
    """The canonical case set lives in code, not in remembered CLI flags."""

    from process_expression_contract import REQUIRED_COVERAGE_CASES

    cases = dict(PROCESSES)
    for name, expression in REQUIRED_COVERAGE_CASES.items():
        assert name not in cases, f"coverage case collides with a process: {name}"
        cases[name] = expression
    rebuilt = build_golden(cases)
    committed = load_golden(GOLDEN_PATH)
    assert rebuilt["golden_sha256"] == committed["golden_sha256"]
    assert len(rebuilt["rows"]) == len(committed["rows"]) == 20


def test_declared_registry_kind_wins_over_the_runtime_value_type():
    """`margin.t` is registered `number`; spelling it `1` must not make it int."""

    assert REGISTRY["margin"].kwargs["t"].kind == "number"
    resolved = resolve_expression("margin(t=1)")
    assert [(k.key, k.value_type, k.value) for k in resolved.kwargs] == [
        ("t", "number", 1)
    ]
    tokens = [t.token for t in token_stream(resolved) if t.token in ("<INT>", "<NUMBER>")]
    assert tokens == ["<NUMBER>"]
    assert "KWARG(t,number)" in role_paths(token_stream(resolved))
    # A genuinely int-declared field is unaffected.
    assert resolve_expression("menu(graph,n=10)").kwargs[0].value_type == "int"


def test_golden_rejects_a_foreign_registry_version():
    document = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    document["registry_version"] = "v0.2"
    with pytest.raises(ContractError, match="different registry version"):
        verify_golden(document)


def test_golden_is_reproducible():
    document = load_golden(GOLDEN_PATH)
    cases = {row["name"]: row["expression"] for row in document["rows"]}
    assert build_golden(cases)["golden_sha256"] == document["golden_sha256"]


# --------------------------------------------------------------------------
# P0 is untouched
# --------------------------------------------------------------------------


def test_p0_behavior_is_unchanged():
    """Step 1 adds contracts; it does not alter existing card behavior."""

    node = pc.parse("lineage(graph,decay=0.85)")
    assert len(pc.ast_sha(node)) == 16
    assert pc.render(node, verbosity=0) == ""
    assert pc.render(node, verbosity=3) == "lineage(graph)"
    assert pc.canonical(node) == "lineage(graph,decay=0.85)"
    assert len(pc.embedding_cache_key(node, 3, "rev")) == 16

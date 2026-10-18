#!/usr/bin/env python3
"""Registry v0.4 test obligations, from ``DESIGN_registry_v0.4.md`` §8.

Per the project convention these land as tests rather than as prose. This file
covers obligations 1-5 and 10-13, which stage 1-3 can discharge; obligations
6-9 belong to the stage-4 migration manifest and are deliberately absent here
— their absence is recorded, not forgotten (§9 sequencing).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import process_cards as pc
from process_cards import (
    ESTIMANDS,
    IMPLS,
    PROCESSES,
    REGISTRY,
    CompositionError,
    DeploymentError,
    ParseError,
    compose_estimands,
    require_deployable,
)

SUBSTRATES = ("pearltrees", "simplemind", "simplewiki", "fs")


# --------------------------------------------------------------------------
# obligation 1: judges fail in substrate positions; substrates pass
# --------------------------------------------------------------------------


@pytest.mark.parametrize("judge", ["haiku", "luna", "sonnet", "graph", "llm"])
def test_obligation_1_lineage_rejects_judges(judge):
    with pytest.raises(ParseError, match="must be substrate"):
        pc.parse(f"lineage({judge})")


@pytest.mark.parametrize("substrate", SUBSTRATES)
def test_obligation_1_lineage_accepts_substrates(substrate):
    node = pc.parse(f"lineage({substrate})")
    assert node.args[0].name == substrate


# --------------------------------------------------------------------------
# obligation 2: every registered corpus parses in a substrate position
# --------------------------------------------------------------------------


@pytest.mark.parametrize("substrate", SUBSTRATES)
@pytest.mark.parametrize("operator", ["lineage", "hop_decay", "lca_frac"])
def test_obligation_2_every_corpus_parses_in_every_substrate_slot(operator, substrate):
    extra = ",gamma=0.6" if operator == "hop_decay" else ""
    node = pc.parse(f"{operator}({substrate}{extra})")
    assert REGISTRY[node.args[0].name].output == "substrate"


# --------------------------------------------------------------------------
# obligation 3: estimand and impl are independent; unknown values refused
# --------------------------------------------------------------------------


def test_obligation_3_estimand_and_impl_are_independently_settable():
    only_estimand = pc.parse('lca_frac(simplemind,estimand="path")')
    only_impl = pc.parse('lca_frac(simplemind,impl="structural")')
    both = pc.parse('lca_frac(simplemind,estimand="path",impl="structural")')
    assert dict(only_estimand.kwargs) == {"estimand": "path"}
    assert dict(only_impl.kwargs) == {"impl": "structural"}
    assert dict(both.kwargs) == {"estimand": "path", "impl": "structural"}


def test_obligation_3_unregistered_impl_is_refused_not_defaulted():
    with pytest.raises(ParseError, match="refused"):
        pc.parse('lca_frac(simplemind,impl="bogus")')
    with pytest.raises(ParseError, match="refused"):
        pc.parse('lca_frac(simplemind,estimand="hop_targets")')  # a procedure, not a relation (R7)


def test_obligation_3_only_existing_implementations_are_registered():
    assert IMPLS == {"structural", "attention"}  # R5: nothing speculative


# --------------------------------------------------------------------------
# obligation 4: absence of estimand blocks deployment; ordinary defaults work
# --------------------------------------------------------------------------


def test_obligation_4_missing_estimand_blocks_deployment():
    with pytest.raises(DeploymentError, match="estimand"):
        require_deployable(pc.parse("lineage(pearltrees,mu=haiku)"))


def test_obligation_4_stating_the_estimand_deploys():
    require_deployable(pc.parse(PROCESSES["lineage-haiku"]))
    require_deployable(pc.parse(PROCESSES["graph-judge"]))


def test_obligation_4_ordinary_registered_defaults_keep_working():
    bare = pc.parse("lineage(pearltrees)")
    assert "decay=0.85" in pc.canonical(bare)  # the default resolves, silently
    # impl= stays optional: it is a selection constraint, not identity (R9).
    require_deployable(pc.parse('lineage(pearltrees,estimand="ancestry")'))


# --------------------------------------------------------------------------
# obligation 5: the registry states decay's direction
# --------------------------------------------------------------------------


def test_obligation_5_decay_and_gamma_are_documented_as_retention():
    for name, key in (("lineage", "decay"), ("hop_decay", "gamma")):
        doc = REGISTRY[name].kwargs[key].doc
        assert "retention" in doc and "per hop" in doc, (name, key)


# --------------------------------------------------------------------------
# obligation 10: adding a pin does not move the semantic digest
# --------------------------------------------------------------------------


def test_obligation_10_pins_move_full_canonical_but_not_the_semantic_digest():
    bare = pc.parse("lineage(pearltrees,decay=0.85)")
    pinned = pc.parse("lineage(pearltrees,decay=0.85)@run/2026-07-25@impl/abc123")
    assert pc.ast_sha(bare) == pc.ast_sha(pinned)
    assert pc.canonical_semantic(bare) == pc.canonical_semantic(pinned)
    assert pc.canonical_full(bare) != pc.canonical_full(pinned)

    from process_identity import full_ast_digest

    assert full_ast_digest(bare) == full_ast_digest(pinned)


def test_obligation_10_holds_for_inner_node_pins_too():
    bare = pc.parse("kalman(luna.D,luna.S)")
    inner = pc.parse("kalman(luna.D@rev/1,luna.S)")
    assert pc.ast_sha(bare) == pc.ast_sha(inner)
    assert pc.canonical_full(bare) != pc.canonical_full(inner)


def test_obligation_10_v3_cache_keys_still_separate_pin_variants():
    """Semantic identity collapses pin variants; the V3 card cache must not,
    because V3 is the verbosity that renders pins."""
    bare = pc.parse("lineage(pearltrees,decay=0.85)")
    pinned = pc.parse("lineage(pearltrees,decay=0.85)@run/2026-07-25")
    assert pc.embedding_cache_key(bare, 3, "rev") != pc.embedding_cache_key(pinned, 3, "rev")
    # V1/V2 elide pins, so the variants render identically and may share keys.
    assert pc.render(bare, 2) == pc.render(pinned, 2)


# --------------------------------------------------------------------------
# obligation 11: the graph judge's registered form, estimand `path` (R8)
# --------------------------------------------------------------------------


def test_obligation_11_graph_judge_registered_form_and_round_trip():
    expression = PROCESSES["graph-judge"]
    node = pc.parse(expression)
    assert node.name == "max"
    assert node.args[0] == 0.02                       # the floor, a literal
    assert node.args[1].name == "product"             # gamma^hops * lca_frac
    inner = node.args[1]
    assert [a.name for a in inner.args] == ["hop_decay", "lca_frac"]
    assert dict(node.kwargs)["estimand"] == "path"
    # Round-trips through render, canonical, and the tokenizer.
    assert pc.parse(pc.render(node, 3)) == node
    assert pc.parse(pc.canonical_full(node)) == node

    from process_expression_tokenizer import assert_round_trips

    assert_round_trips(expression)


def test_obligation_11_components_share_the_substrate_by_spelling_only():
    """Known v0.4 limitation, recorded as designed behavior (§10.1): the flat
    signatures cannot check that hop_decay and lca_frac walk one substrate.
    A mismatched pair parses — legal-but-odd; vNext index unification is the
    layer that checks it."""
    mismatched = pc.parse("product(hop_decay(simplemind,gamma=0.6),lca_frac(fs))")
    assert mismatched is not None


# --------------------------------------------------------------------------
# obligation 12: estimand enumeration and the composition typing rule (R7)
# --------------------------------------------------------------------------


def test_obligation_12_estimand_values_are_exactly_the_r7_enumeration():
    assert ESTIMANDS == {
        "subcategory", "super_category", "element_of", "subtopic",
        "see_also", "assoc", "bridge", "ancestry", "path",
    }
    for value in sorted(ESTIMANDS):
        pc.parse(f'lca_frac(fs,estimand="{value}")')


def test_obligation_12_derived_names_come_from_the_composition_rule():
    # Monotone descent composes to ancestry …
    assert compose_estimands(["subcategory", "subcategory"]) == "ancestry"
    # … and any mixed-direction composition types as path (sibling shape).
    assert compose_estimands(["subcategory", "super_category"]) == "path"
    assert compose_estimands(["subtopic", "super_category", "super_category"]) == "path"
    # bridge is transparent to typing.
    assert compose_estimands(["subcategory", "bridge", "subcategory"]) == "ancestry"
    assert compose_estimands(["bridge"]) == "bridge"
    # A single primitive composes to itself, not to a derived name.
    assert compose_estimands(["subcategory"]) == "subcategory"


@pytest.mark.parametrize("chain", [
    ["assoc"],
    ["see_also", "subcategory"],
    ["subcategory", "assoc", "subcategory"],
])
def test_obligation_12_assoc_and_see_also_inside_a_chain_are_type_errors(chain):
    with pytest.raises(CompositionError, match="excluded from chains"):
        compose_estimands(chain)


def test_obligation_12_derived_estimands_are_not_chain_steps():
    with pytest.raises(CompositionError, match="primitive"):
        compose_estimands(["ancestry", "subcategory"])


# --------------------------------------------------------------------------
# obligation 13: subtopic/subcategory interchange; element_of at the item end
# --------------------------------------------------------------------------


def test_obligation_13_subtopic_and_subcategory_are_interchangeable_in_chains():
    for chain in (
        ["subtopic", "subcategory"],
        ["subcategory", "subtopic"],
        ["subtopic", "subtopic"],
    ):
        assert compose_estimands(chain) == "ancestry"
    # With membership at the item end the composite stays element_of:
    # membership survives descent (element_of ∘ subcategory ⇒ element_of).
    assert compose_estimands(["element_of", "subcategory"]) == "element_of"
    assert compose_estimands(["element_of", "subtopic", "subcategory"]) == "element_of"


@pytest.mark.parametrize("chain", [
    ["subcategory", "element_of"],
    ["element_of", "element_of"],
    ["subtopic", "element_of", "subcategory"],
])
def test_obligation_13_element_of_is_rejected_anywhere_but_the_item_end(chain):
    with pytest.raises(CompositionError, match="item end"):
        compose_estimands(chain)


# --------------------------------------------------------------------------
# stage-1 blueprint spot checks (§10.2): the entry table is what landed
# --------------------------------------------------------------------------


def test_source_type_is_retired_and_the_split_is_total():
    outputs = {s.output for s in REGISTRY.values()}
    assert "source" not in outputs
    assert {"substrate", "judge"} <= outputs


def test_graph_is_judge_only():
    assert REGISTRY["graph"].output == "judge"
    with pytest.raises(ParseError):
        pc.parse("lineage(graph)")  # its former substrate role is retired


def test_mu_takes_a_judge_expression_and_not_a_substrate():
    node = pc.parse("lineage(pearltrees,mu=sonnet.lineage)")
    assert dict(node.kwargs)["mu"].mods == ("lineage",)
    with pytest.raises(ParseError, match="judge expression"):
        pc.parse("lineage(pearltrees,mu=fs)")
    with pytest.raises(ParseError, match="judge expression"):
        pc.parse("lineage(pearltrees,mu=0.5)")


def test_positional_literals_are_typed_by_the_declared_kind():
    pc.parse("max(0.02,e5(margin(t=0.03)))")          # number|score admits a number
    with pytest.raises(ParseError, match="must be"):
        pc.parse("product(0.02,lca_frac(fs))")        # score-only slot refuses it
    with pytest.raises(ParseError, match="must be"):
        pc.parse("max(0.02,0.03)")                    # variadic slots are score-only


def test_lineage_graph_is_retired_from_the_process_registry():
    assert "lineage-graph" not in PROCESSES
    assert {"graph-judge", "lineage-haiku"} <= set(PROCESSES)
    assert len(PROCESSES) == 10

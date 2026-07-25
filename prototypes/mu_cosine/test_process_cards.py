"""P0 acceptance tests: parse-the-spec, round-trip, identity/card split, determinism."""
import pytest

from process_cards import (
    PROCESSES,
    REGISTRY,
    REGISTRY_VERSION,
    RENDERER_VERSION,
    Node,
    ParseError,
    ast_sha,
    canonical,
    embedding_cache_key,
    parse,
    render,
    validate,
)


def test_registry_parses_every_registered_process():
    for name, expr in PROCESSES.items():
        node = parse(expr)
        assert parse(render(node, 3)) == node, name          # round-trip at V3


def test_registry_parses_all_three_spec_cards():
    examples = [
        "e5(routing(e5, sonnet))",
        "e5(routing(e5, sonnet.lineage, t=[0.02,0.03], menus=[10,20]))",
        (
            'e5(routing(e5@e5-small-v2, '
            'sonnet.lineage@2026-07-23/menu-order-blind, '
            't=[0.02,0.03], menus=[10,20], manifest="fcf5e1d6"))'
        ),
    ]
    for expression in examples:
        node = parse(expression)
        assert parse(render(node, 3)) == node


def test_dotted_and_hyphen_dot_names():
    n = parse("kalman(luna.D,luna.S)")
    assert n.args[0].mods == ("D",)                          # uppercase channel modifier
    assert parse("gpt-5.5-low").name == "gpt-5.5-low"        # dots inside registered name


def test_canonical_resolves_defaults_and_is_lossless():
    a = parse("lineage(graph)")
    b = parse("lineage(graph,decay=0.85)")
    assert canonical(a) == canonical(b) and ast_sha(a) == ast_sha(b)
    assert render(a, 1) == "lineage(graph)"                  # V1 elides kwargs entirely
    assert "decay=0.85" in canonical(a)                      # identity keeps resolved defaults


def test_verbosity_ladder_monotone():
    n = parse("e5(routing(e5,sonnet.lineage,t=[0.02,0.03],menus=[10,20]))@fcf5e1d6")
    v = [render(n, k) for k in (0, 1, 2, 3)]
    assert v[0] == "" and "t=" not in v[1] and "t=[0.02,0.03]" in v[2] and "@fcf5e1d6" in v[3]
    assert "@" not in v[2]                                   # pins are V3-only


def test_cache_key_binds_revision_and_verbosity():
    n = parse("kalman(luna.D,luna.S)")
    k = embedding_cache_key(n, 1, "rev-a")
    assert k != embedding_cache_key(n, 2, "rev-a")
    assert k != embedding_cache_key(n, 1, "rev-b")
    assert k == embedding_cache_key(parse("kalman(luna.D,luna.S)"), 1, "rev-a")  # deterministic


def test_fail_closed():
    with pytest.raises(ParseError):
        parse("mystery(e5)")                                 # unregistered operator
    with pytest.raises(ParseError):
        parse("routing")                                     # operator used as atom
    with pytest.raises(ParseError):
        parse("human(e5)")                                   # atom used as operator


@pytest.mark.parametrize(
    "expression",
    [
        "e5()",                                              # arity
        "kalman()",                                          # arity
        "kalman(margin(t=0.1),luna.S)",                      # positional type
        "lineage(graph,bogus=1)",                            # unknown kwarg
        "lineage(graph,depth=1,depth=2)",                    # duplicate kwarg
        "lineage(graph,depth=1.5)",                          # wrong kwarg type
        "routing(e5,sonnet,t=[0.02],menus=[10.5])",          # wrong list type
        "routing(e5,sonnet,t=[],menus=[10])",                # empty typed list
        "routing(e5,sonnet,t=[0.02])",                       # paired kwargs
        "routing(e5,sonnet,t=[0.02],menus=[10,20])",         # paired lengths
        "blend(luna.D,luna.S,w=[1])",                        # one weight per source
        "sonnet.unknown",                                    # unknown modifier
        "luna.D.S",                                          # mutually exclusive modifiers
        "luna.D.D",                                          # duplicate modifier
        "routing(e5,sonnet,t=[0.02,],menus=[10])",           # trailing list comma
        "routing(e5,sonnet,t=[0.02],menus=[10],)",           # trailing call comma
    ],
)
def test_typed_registry_rejects_malformed_expressions(expression):
    with pytest.raises(ParseError):
        parse(expression)


def test_strings_and_whitespace_round_trip_without_content_loss():
    expression = (
        'e5(routing(e5, sonnet.lineage, t = [0.02, 0.03], '
        'menus = [10, 20], manifest = "hash with \\"quoted\\" space"))'
    )
    node = parse(expression)
    rendered = render(node, 3)
    assert '"hash with \\"quoted\\" space"' in rendered
    assert parse(rendered) == node


def test_float_canonicalization_is_lossless_and_collision_resistant():
    first = parse("lineage(graph,decay=0.12345671)")
    second = parse("lineage(graph,decay=0.12345672)")
    assert canonical(first) != canonical(second)
    assert ast_sha(first) != ast_sha(second)
    for expression in (
        "lineage(graph,decay=1.0000001)",
        "lineage(graph,decay=1234567.1)",
        "lineage(graph,decay=1e-12)",
    ):
        node = parse(expression)
        assert parse(render(node, 3)) == node
        assert parse(canonical(node)) == node


def test_nonfinite_numeric_input_fails_closed():
    with pytest.raises(ParseError, match="non-finite"):
        parse("lineage(graph,decay=1e9999)")


def test_invalid_unicode_string_fails_before_identity_hashing():
    with pytest.raises(ParseError, match="Unicode surrogate"):
        parse('routing(e5,sonnet,manifest="\\ud800")')


@pytest.mark.parametrize("verbosity", [-1, 4, True, 1.5])
def test_invalid_verbosity_fails_closed(verbosity):
    node = parse("kalman(luna.D,luna.S)")
    with pytest.raises(ValueError, match="verbosity"):
        render(node, verbosity)
    with pytest.raises(ValueError, match="verbosity"):
        embedding_cache_key(node, verbosity, "rev-a")


def test_programmatic_nodes_are_validated_before_render_or_hash():
    malformed = Node("kalman", (Node("human"),))
    with pytest.raises(ParseError, match="expects"):
        validate(malformed)
    with pytest.raises(ParseError, match="expects"):
        canonical(malformed)
    with pytest.raises(ParseError, match="expects"):
        ast_sha(malformed)
    with pytest.raises(ParseError, match="unregistered"):
        validate(Node([], ()))
    with pytest.raises(ParseError, match="malformed modifier"):
        validate(Node("luna", mods=([],)))
    with pytest.raises(ParseError, match="must be string"):
        ast_sha(
            Node(
                "routing",
                (Node("e5"), Node("sonnet")),
                (("manifest", "\ud800"),),
            )
        )


def test_registry_is_versioned_and_has_typed_signatures():
    assert REGISTRY_VERSION == "v0.3"
    assert RENDERER_VERSION == "r2"
    assert REGISTRY["routing"].arg_types == ("score", "source")
    assert REGISTRY["routing"].kwargs["manifest"].kind == "string"
    assert REGISTRY["routing"].output == "pick"

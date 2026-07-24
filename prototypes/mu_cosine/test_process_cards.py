"""P0 acceptance tests: parse-the-spec, round-trip, identity/card split, determinism."""
import pytest

from process_cards import (PROCESSES, ParseError, ast_sha, canonical,
                           embedding_cache_key, parse, render)


def test_registry_parses_every_registered_process():
    for name, expr in PROCESSES.items():
        node = parse(expr)
        assert parse(render(node, 3)) == node, name          # round-trip at V3


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

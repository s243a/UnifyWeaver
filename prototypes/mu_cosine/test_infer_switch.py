#!/usr/bin/env python3
"""Regression tests for the inferred-relation stochastic operator switch (PR #3356 review asks):
TAGGED rows never switch, legacy 9-column pairs default to conf=1.0, the switch is seed-reproducible, and its
RNG is isolated. Run: `python3 test_infer_switch.py` — plain asserts, no pytest. (NOT dependency-free: it
imports `train_mu_attention`, which imports torch; the functions under test, `switched_op`/`load_graded`,
are themselves pure Python.)"""
import os
import random
import tempfile

from train_mu_attention import switched_op, load_graded


def test_tagged_never_switches():
    rng = random.Random(0)
    assert all(switched_op("ELEM", "element_of", 1.0, 999, 0.4, 20, rng) == "ELEM" for _ in range(2000))


def test_non_element_never_switches():
    rng = random.Random(0)
    assert switched_op("SYM", "bridge", 0.4, 999, 0.4, 20, rng) == "SYM"
    assert switched_op("WIKI", "subcategory", 0.4, 999, 0.4, 20, rng) == "WIKI"


def test_inferred_switches_at_expected_rate():
    # p = base·min(1, breadth/scale)·(1−conf) = 0.4 · min(1,100/20) · (1−0.4) = 0.4·1·0.6 = 0.24
    rng = random.Random(1)
    n = sum(switched_op("ELEM", "element_of", 0.4, 100, 0.4, 20, rng) == "WIKI" for _ in range(4000))
    assert 0.20 < n / 4000 < 0.28, n / 4000


def test_confidence_scales_probability():
    # higher confidence ⇒ lower switch rate (0.8 inferred switches far less than 0.4)
    r1, r2 = random.Random(2), random.Random(2)
    lo = sum(switched_op("ELEM", "element_of", 0.4, 100, 0.4, 20, r1) == "WIKI" for _ in range(4000))
    hi = sum(switched_op("ELEM", "element_of", 0.8, 100, 0.4, 20, r2) == "WIKI" for _ in range(4000))
    assert hi < lo


def test_seed_reproducible():
    r1, r2 = random.Random(7), random.Random(7)
    s1 = [switched_op("ELEM", "element_of", 0.4, 50, 0.4, 20, r1) for _ in range(1000)]
    s2 = [switched_op("ELEM", "element_of", 0.4, 50, 0.4, 20, r2) for _ in range(1000)]
    assert s1 == s2


def _write_pairs(cols):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "g_pairs.tsv")
    with open(p, "w") as f:
        f.write("# header\n")
        f.write("\t".join(cols) + "\n")
    return p


def test_legacy_9col_defaults_conf_1():
    p = _write_pairs(["a", "b", "0.90", "ELEM", "element_of", "page", "pearltrees_collection", "pt", "human"])
    rows, _ = load_graded(p)
    assert len(rows[0]) == 10 and rows[0][9] == 1.0    # legacy 9-col ⇒ conf defaults to 1.0 (tagged)


def test_10col_reads_conf():
    p = _write_pairs(["a", "b", "0.90", "ELEM", "element_of", "page", "pt_collection", "pt", "human", "0.40"])
    rows, _ = load_graded(p)
    assert rows[0][9] == 0.4


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"all {len(tests)} infer-switch tests passed")

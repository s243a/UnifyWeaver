#!/usr/bin/env python3
"""Pure-function tests for the structured residual covariance campaign runner."""
import copy
from types import SimpleNamespace

import numpy as np
import torch
import fine_tune_channel_heads as channel_heads

from run_structured_residual_covariance import (
    aggregate_gate,
    decision_metrics,
    file_provenance,
    semantic_pair_features,
    state_metrics,
    train_standardize,
)


def test_file_provenance_hashes_exact_regular_file(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"structured-covariance\n")
    got = file_provenance(artifact)
    assert got["path"] == str(artifact)
    assert got["size_bytes"] == 22
    assert got["sha256"] == "f8eac5833bfc8255254a47f92350e9ebce1f5ec59ed27741f13a6a9af27f2a5d"


def test_campaign_loader_honors_explicit_scored_path(tmp_path, monkeypatch):
    scored = tmp_path / "custom_campaign.tsv"
    scored.write_text(
        "#node\troot\tneighborhood\tmu[subcategory]\tmu[subtopic]\tmu[element_of]"
        "\tmu[super_category]\tmu[see_also]\tmu[assoc]\n"
        "custom-node\tcustom-root\tcustom-tag\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\n",
        encoding="utf-8",
    )
    parents = {"custom-node": ("custom-root",), "custom-root": ()}
    children = {"custom-root": ("custom-node",)}
    monkeypatch.setattr(
        channel_heads,
        "load_feature_graph",
        lambda config: (parents, children, {"custom-node": 1, "custom-root": 1}, None),
    )
    monkeypatch.setattr(
        channel_heads,
        "load_e5_cache_and_filter",
        lambda pairs, hop, D, S, cache: (
            {"query": {}, "passage": {}}, {}, pairs, hop, D, S
        ),
    )
    monkeypatch.setattr(channel_heads, "Tokenizer", lambda *args: "tokenizer")
    monkeypatch.setattr(channel_heads, "hit_prob", lambda *args: 1.0)
    monkeypatch.setattr(
        channel_heads,
        "descendant_disjoint_split",
        lambda pairs, seed, held_frac: (np.array([0]), np.array([], dtype=int)),
    )
    loaded = channel_heads.load_campaign_datasets(campaign_scored=scored)
    assert loaded["exploratory-campaign"]["pairs"] == [("custom-node", "custom-root")]
    assert loaded["fresh-campaign"]["tags"] == ["custom-tag"]


def test_semantic_pair_features_are_item_aligned_normalized_and_role_aware():
    tok = SimpleNamespace(
        idx={"a": 0, "b": 1, "r": 2},
        p=torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        q=torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 3.0]]),
    )
    got = semantic_pair_features({"tok": tok}, [("a", "r"), ("b", "r")])
    assert got.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0)
    assert not np.allclose(got[0], got[1])
    # Candidate uses passage coordinates; root uses query coordinates.
    np.testing.assert_allclose(got[0] / got[0, 0], [1.0, 0.0, 0.0, 3.0])


def test_train_standardizer_does_not_fit_on_held_rows():
    features = np.array([[0.0, 1.0], [2.0, 3.0], [10.0, -20.0], [30.0, 40.0]])
    train = np.array([0, 1])
    first, mean1, scale1 = train_standardize(features, train)
    changed = features.copy()
    changed[2:] *= 1e6
    second, mean2, scale2 = train_standardize(changed, train)
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(scale1, scale2)
    np.testing.assert_array_equal(first[train], second[train])
    np.testing.assert_allclose(first[train].mean(axis=0), 0.0, atol=1e-15)


def test_state_metrics_match_known_standard_normal_geometry():
    observed = np.array([[1.0, 0.0], [0.0, 2.0]])
    mean = np.zeros_like(observed)
    covariance = np.repeat(np.eye(2)[None], 2, axis=0)
    got = state_metrics(observed, mean, covariance)
    assert np.isclose(got["mse_per_scalar"], 1.25)
    assert np.isclose(got["mahalanobis_per_dimension"], 1.25)
    assert got["coverage_95_bivariate_ellipse"] == 1.0
    expected = np.mean([0.5 * (1.0 + 2 * np.log(2 * np.pi)),
                        0.5 * (4.0 + 2 * np.log(2 * np.pi))])
    assert np.isclose(got["mean_bivariate_nll"], expected)


def test_decision_metrics_use_margin_gate_and_equal_width_ece():
    proba = np.array([
        [0.90, 0.05, 0.05],
        [0.10, 0.80, 0.10],
        [0.33, 0.33, 0.34],
        [0.15, 0.15, 0.70],
    ])
    got = decision_metrics(proba, ["directional", "symmetric", "other", "other"])
    assert got["accuracy"] == 1.0
    assert got["log_loss"] > 0.0
    assert 0.0 <= got["ece_10_equal_width"] <= 1.0
    assert got["aurc_margin"] == 0.0


def _gate_row(corpus, seed, block, separable, dense):
    def metric(value, *, loading=0.0):
        return {
            "held_joint_residual_nll_per_scalar": value,
            "posterior": {
                "state": {"mean_bivariate_nll": value},
                "decision": {"log_loss": value, "aurc_margin": value},
                "conditioner": {
                    "loading": {"relative_diagonal_loading": loading},
                    "prior_loading": {"relative_diagonal_loading": 0.0},
                },
            },
        }

    return {
        "corpus": corpus,
        "seed": seed,
        "models": {
            "block_global": metric(block + 0.05),
            "block_regional": metric(block),
            "separable_regional": metric(separable),
            "dense_lmc_regional": metric(dense),
        },
    }


def test_engineering_gate_requires_all_ten_predeclared_seeds():
    smoke = [_gate_row(corpus, seed, 1.0, 0.9, 0.8)
             for corpus in ("exploratory", "fresh") for seed in range(2)]
    assert not aggregate_gate(smoke)["gate_evaluable"]

    primary = [_gate_row(corpus, seed, 1.0, 0.85, 0.8)
               for corpus in ("exploratory", "fresh") for seed in range(10)]
    gate = aggregate_gate(primary)
    assert gate["gate_evaluable"]
    assert gate["dense_lmc_passes_direction_and_stability"]
    assert gate["separable_passes_direction_and_stability"]
    assert np.isclose(gate["macro_split_separable_fraction_of_dense_gain"], 0.75)
    assert not gate["passes_80_percent_recovery"]

    duplicated = [_gate_row(corpus, 0, 1.0, 0.85, 0.8)
                  for corpus in ("exploratory", "fresh") for _ in range(10)]
    assert not aggregate_gate(duplicated)["gate_evaluable"]

    one_corpus = [_gate_row("fresh", seed, 1.0, 0.85, 0.8) for seed in range(10)]
    assert not aggregate_gate(one_corpus)["gate_evaluable"]


def test_engineering_gate_pins_direction_mean_and_every_guardrail():
    passing = [_gate_row(corpus, seed, 1.0, 0.8, 0.8)
               for corpus in ("exploratory", "fresh") for seed in range(10)]
    assert aggregate_gate(passing)["structured_covariance_gate_passes"]

    exactly_eight = [
        _gate_row(corpus, seed, 1.0, 0.9 if seed < 8 else 1.01,
                  0.9 if seed < 8 else 1.01)
        for corpus in ("exploratory", "fresh") for seed in range(10)
    ]
    gate = aggregate_gate(exactly_eight)
    assert gate["dense_lmc_passes_direction_and_stability"]
    assert gate["separable_passes_direction_and_stability"]

    negative_mean = [
        _gate_row(corpus, seed, 1.0, 0.99 if seed < 8 else 1.10,
                  0.99 if seed < 8 else 1.10)
        for corpus in ("exploratory", "fresh") for seed in range(10)
    ]
    gate = aggregate_gate(negative_mean)
    assert all(value["separable_positive_seeds"] == 8 for value in gate["by_corpus"].values())
    assert not gate["separable_passes_direction_and_stability"]

    bad_posterior = copy.deepcopy(passing)
    for row in bad_posterior:
        if row["corpus"] == "fresh":
            baseline = row["models"]["block_regional"]["posterior"]["state"]
            row["models"]["separable_regional"]["posterior"]["state"][
                "mean_bivariate_nll"
            ] = baseline["mean_bivariate_nll"] + 1e-4
    assert not aggregate_gate(bad_posterior)["separable_posterior_nll_guardrail_passes"]

    decision_boundary = copy.deepcopy(passing)
    for row in decision_boundary:
        baseline = row["models"]["block_regional"]["posterior"]["decision"]
        candidate = row["models"]["separable_regional"]["posterior"]["decision"]
        candidate["log_loss"] = baseline["log_loss"] + 0.009999
        candidate["aurc_margin"] = baseline["aurc_margin"] + 0.009999
    assert aggregate_gate(decision_boundary)["separable_decision_guardrail_passes"]
    for row in decision_boundary:
        if row["corpus"] == "fresh":
            baseline = row["models"]["block_regional"]["posterior"]["decision"]
            row["models"]["separable_regional"]["posterior"]["decision"][
                "log_loss"
            ] = baseline["log_loss"] + 0.010001
    assert not aggregate_gate(decision_boundary)["separable_decision_guardrail_passes"]

    bad_loading = copy.deepcopy(passing)
    bad_loading[0]["models"]["dense_lmc_regional"]["posterior"]["conditioner"][
        "prior_loading"
    ]["relative_diagonal_loading"] = 0.0011
    assert not aggregate_gate(bad_loading)["loading_budget_guardrail_passes"]

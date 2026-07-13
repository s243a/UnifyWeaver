import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_repeated_judge_power as runner
from run_repeated_judge_power import (
    FULL_CONFIGURATIONS,
    FULL_SCENARIOS,
    _configuration_decision,
    _one_sided_exact_binomial_pvalue,
    build_arg_parser,
    build_scientific_payload,
    main,
    parse_configuration,
    resolve_configuration,
    serialize_payload,
)


def test_configuration_parser_and_full_preregistration_contract_are_exact():
    assert parse_configuration("160:3") == (160, 3)
    with pytest.raises(Exception):
        parse_configuration("160")
    resolved = resolve_configuration(build_arg_parser().parse_args(["--full-prereg"]))
    assert resolved["configurations"] == FULL_CONFIGURATIONS
    assert resolved["null_draws"] == 1999
    assert resolved["power_replicates"] == 200
    assert resolved["multiplier_draws"] == 999
    assert resolved["confidence"] == 0.95
    assert resolved["outer_folds"] == 5
    assert resolved["inner_folds"] == 3
    assert resolved["mean_ridges"] == (0.0, 0.01, 0.1, 1.0, 10.0)
    assert resolved["gammas"] == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert resolved["rhos"] == (0.0, 0.025, 0.05, 0.1, 0.2)
    assert resolved["scenarios"] == FULL_SCENARIOS
    assert resolved["prompt_batch_rows"] == 10
    assert (320, 4) in FULL_CONFIGURATIONS
    assert (800, 4) in FULL_CONFIGURATIONS
    assert (160, 4) not in FULL_CONFIGURATIONS


@pytest.mark.parametrize(
    "override",
    [
        ["--config", "32:3"],
        ["--null-draws", "1999"],
        ["--power-replicates", "200"],
        ["--confidence", ".95"],
        ["--outer-folds", "5"],
        ["--mean-ridges", "0", ".01", ".1", "1", "10"],
        ["--gammas", "0", ".25", ".5", ".75", "1"],
        ["--rhos", "0", ".025", ".05", ".1", ".2"],
        ["--scenarios", "block_null"],
        ["--prompt-batch-rows", "10"],
        ["--seed", "1207000"],
    ],
)
def test_full_prereg_rejects_every_scientific_override_even_if_value_matches(override):
    args = build_arg_parser().parse_args(["--full-prereg", *override])
    with pytest.raises(ValueError, match="rejects scientific overrides"):
        resolve_configuration(args)


def test_nonfull_mode_fails_closed_on_unsupported_folds_or_prompt_capacity():
    parser = build_arg_parser()
    with pytest.raises(ValueError, match="exactly five outer"):
        resolve_configuration(parser.parse_args(["--outer-folds", "4"]))
    with pytest.raises(ValueError, match=r"\[1,10\]"):
        resolve_configuration(parser.parse_args(["--prompt-batch-rows", "11"]))


def fake_scenario_row(name, *, promotions=180, gain=0.02, topology=0.90):
    replicates = 200
    if name == "block_null":
        promotions = 6
        gain = 0.0
        topology = None
    elif name == "mean_only":
        promotions = 0
        gain = 0.0
        topology = None
    elif name.startswith("deranged"):
        topology = None
    return {
        "scenario": name,
        "replicates": replicates,
        "joint_synthetic_primary_events": promotions,
        "joint_synthetic_primary_event_rate": promotions / replicates,
        "endpoint_mean_gain_per_scalar": {
            corpus: {
                "residual_nll": {"mean": gain},
                "posterior_state_nll": {"mean": gain},
            }
            for corpus in ("synthetic_corpus_1", "synthetic_corpus_2")
        },
        "topology_truth_beats_derangement_rate": topology,
    }


def passing_rows():
    return [fake_scenario_row(name) for name in FULL_SCENARIOS]


def test_exact_binomial_and_complete_sizing_decision():
    assert _one_sided_exact_binomial_pvalue(0, 200) == 1.0
    assert _one_sided_exact_binomial_pvalue(6, 200) > 0.05
    assert _one_sided_exact_binomial_pvalue(20, 200) < 0.05
    decision = _configuration_decision(passing_rows(), FULL_SCENARIOS)
    assert decision["evaluable"] is True
    assert decision["pass"] is True
    assert decision["block_null"][
        "does_not_reject_p_at_most_0_05_and_observed_at_most_0_10"
    ] is True


def test_decision_fails_on_incomplete_grid_or_either_primary_endpoint():
    incomplete = _configuration_decision(
        [fake_scenario_row("block_null")], ["block_null"]
    )
    assert incomplete == {
        "evaluable": False,
        "pass": False,
        "reason": "the complete frozen scenario grid is required",
    }
    rows = passing_rows()
    target = next(row for row in rows if row["scenario"] == "nomic_rho_0.10")
    target["endpoint_mean_gain_per_scalar"]["synthetic_corpus_2"][
        "posterior_state_nll"
    ]["mean"] = -0.001
    assert _configuration_decision(rows, FULL_SCENARIOS)["pass"] is False


def test_smallest_r3_passing_G_is_recommendation_and_r4_stays_sensitivity(monkeypatch):
    resolved = resolve_configuration(build_arg_parser().parse_args([]))
    resolved["configurations"] = ((160, 3), (320, 3), (320, 4), (800, 4))
    resolved["scenarios"] = FULL_SCENARIOS

    def fake_configuration(configuration, _resolved):
        components, repeats = configuration
        passed = components >= 320 and repeats == 3
        return {
            "components_per_required_corpus": components,
            "repeats": repeats,
            "decision": {"evaluable": True, "pass": passed},
        }

    monkeypatch.setattr(runner, "run_configuration", fake_configuration)
    payload = build_scientific_payload(resolved)
    assert payload["gate"]["smallest_passing_synthetic_primary_event_G_per_corpus"] == 320
    assert payload["gate"]["synthetic_primary_event_pass"] is True
    assert payload["gate"]["final_campaign_G_recommendation"] is None
    assert payload["gate"]["deployment_pass"] is False
    assert [row["repeats"] for row in payload["primary_r3_results"]] == [3, 3]
    assert [row["repeats"] for row in payload["repeat4_sensitivity_results"]] == [4, 4]
    assert payload["gate"]["real_covariance_deployment"] is False


def tiny_resolved():
    return resolve_configuration(build_arg_parser().parse_args([
        "--config", "32:3",
        "--null-draws", "1",
        "--power-replicates", "1",
        "--multiplier-draws", "7",
        "--mean-ridges", "0", ".1",
        "--gammas", "0",
        "--rhos", "0",
        "--scenarios", "block_null",
        "--missing-rate", "0",
        "--seed", "81000",
    ]))


def test_small_scientific_payload_is_deterministic_and_provenanced():
    resolved = tiny_resolved()
    first = build_scientific_payload(resolved)
    second = build_scientific_payload(resolved)
    assert serialize_payload(first) == serialize_payload(second)
    assert first["gate"]["real_covariance_deployment"] is False
    assert first["gate"]["synthetic_primary_event_sizing_evaluable"] is False
    assert first["primary_r3_results"][0]["synthetic_joint_selector_null"]["draws"] == 1
    assert "wall_seconds" not in serialize_payload(first)
    assert "/tmp/" not in serialize_payload(first)
    assert "path" not in first["provenance"]["files"]["power_runner"]
    for record in first["provenance"]["files"].values():
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64
    scenario = first["primary_r3_results"][0]["scenarios"][0]
    assert scenario["inference_prompt_blocks"]["synthetic_corpus_1"]["maximum"] < 32
    assert set(scenario["endpoint_mean_gain_per_scalar"]) == {
        "synthetic_corpus_1", "synthetic_corpus_2"
    }


def test_cli_atomically_creates_parent_and_output_is_path_independent(tmp_path):
    common = [
        "--config", "32:3",
        "--null-draws", "1",
        "--power-replicates", "1",
        "--multiplier-draws", "7",
        "--mean-ridges", "0", ".1",
        "--gammas", "0",
        "--rhos", "0",
        "--scenarios", "block_null",
        "--missing-rate", "0",
        "--seed", "91000",
    ]
    first_path = tmp_path / "new" / "first.json"
    second_path = tmp_path / "other" / "second.json"
    main(common + ["--out", str(first_path)])
    main(common + ["--out", str(second_path)])
    assert first_path.read_bytes() == second_path.read_bytes()
    assert not list(tmp_path.rglob("*.tmp"))
    payload = json.loads(first_path.read_text())
    assert payload["gate"]["synthetic_primary_event_sizing_evaluable"] is False

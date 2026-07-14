import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_repeated_judge_source_power as runner
from repeated_judge_power import BLOCK_CANDIDATE
from repeated_judge_source_power import SourceCorpusPowerReplicate, SourcePowerReplicate


def load_bundle():
    return runner.load_source_design_bundle(
        runner.DEFAULT_SOURCE_DESIGN, runner.DEFAULT_SUMMARY_PATH
    )


def exact_resolved():
    return runner.resolve_configuration(
        runner.build_arg_parser().parse_args(["--full-prereg"])
    )


def tiny_resolved():
    return runner.resolve_configuration(
        runner.build_arg_parser().parse_args([
            "--config", "64:160:3",
            "--source-eta-grid", "0",
            "--null-types", "block_null",
            "--scenarios", "block_null",
            "--null-draws", "2",
            "--power-replicates", "2",
            "--multiplier-draws", "3",
            "--gammas", ".5",
            "--item-rhos", "0",
            "--mean-ridges", "0", ".1",
            "--missing-rate", "0",
            "--seed", "81000",
        ])
    )


def fake_aggregate_row(
    scenario, exploratory_source_eta, fresh_source_eta, *, events=None,
    replicates=200,
):
    control = scenario in ("block_null", "mean_only")
    deranged = scenario.startswith("deranged")
    if events is None:
        events = 0 if control or deranged else replicates
    gain = 0.0 if control or deranged else 0.1
    topology = None if control or deranged else 1.0
    return {
        "scenario": scenario,
        "generator_source_eta_by_corpus": {
            "exploratory": float(exploratory_source_eta),
            "fresh": float(fresh_source_eta),
        },
        "replicates": replicates,
        "both_corpus_selector_rejections": events,
        "both_corpus_selector_rejection_rate": events / replicates,
        "joint_source_primary_events": events,
        "joint_source_primary_event_rate": events / replicates,
        "inference_nonidentified_replicates": 0,
        "all_replicates_inference_identified": True,
        "endpoint_mean_gain_per_scalar": {
            corpus: {
                endpoint: {
                    "mean": gain,
                    "sd": 0.0,
                    "minimum": gain,
                    "maximum": gain,
                }
                for endpoint in runner.PRIMARY_ENDPOINTS
            }
            for corpus in runner.REQUIRED_SOURCE_CORPORA
        },
        "worst_source_eta_simultaneous_lower_bound": {},
        "inference_source_eta_grid": list(runner.SOURCE_ETA_GRID),
        "multiplier_max_t_critical_value": {
            "mean": 1.0, "sd": 0.0, "minimum": 1.0, "maximum": 1.0,
        },
        "topology_truth_beats_derangement_rate": topology,
        "selected_gamma_counts_across_outer_folds": {"0.5": 10},
        "selected_rho_item_counts_across_outer_folds": {"0.0": 10},
        "maximum_call_loading": 0.0,
        "maximum_persistent_loading": 0.0,
        "maximum_request_loading": 0.0,
    }


def complete_fake_rows(*, events=200):
    rows = []
    for scenario in runner.FULL_SCENARIOS:
        for exploratory_source_eta, fresh_source_eta in runner._source_eta_pairs(
            runner.SOURCE_ETA_GRID
        ):
            value = None
            if scenario in runner.PRIMARY_TRUTH_SCENARIOS:
                value = events
            rows.append(fake_aggregate_row(
                scenario, exploratory_source_eta, fresh_source_eta,
                events=value,
            ))
    return rows


def fake_power_record(
    scenario="block_null", exploratory_source_eta=0.0, fresh_source_eta=0.0
):
    truth = scenario not in ("block_null", "mean_only") and not scenario.startswith(
        "deranged"
    )
    selected = tuple(
        tuple(BLOCK_CANDIDATE for _ in range(5)) for _ in range(2)
    )
    gain = 0.1 if truth else 0.0
    return SourcePowerReplicate(
        scenario=scenario,
        generator_source_eta_by_corpus=(
            float(exploratory_source_eta), float(fresh_source_eta)
        ),
        corpus_names=runner.REQUIRED_SOURCE_CORPORA,
        selected=selected,
        corpus_selector_rejected=(truth, truth),
        familywise_rejected=truth,
        maximum_inner_gain=gain,
        endpoint_component_gains=np.empty((0, 0, 0)),
        endpoint_mean_gains=np.full((2, 2), gain),
        inference_source_eta_grid=(0.0,),
        endpoint_lower_bounds_by_source_eta=np.empty((0, 0, 0)),
        endpoint_worst_source_eta_lower_bounds=np.full((2, 2), gain),
        inference_identified=True,
        multiplier_critical_value=1.0,
        multiplier_order_statistic_rank_one_based=3,
        inference_prompt_blocks=(16, 16),
        inference_source_regions=(40, 64),
        topology_component_advantage=None,
        topology_truth_beats_derangement=True if truth else None,
        promoted=truth,
        call_loading=0.0,
        persistent_loading=0.0,
        request_loading=0.0,
    )


def fake_corpus_power_record(corpus, scenario, source_eta, components=160):
    truth = scenario not in ("block_null", "mean_only") and not scenario.startswith(
        "deranged"
    )
    gain = 0.1 if truth else 0.0
    return SourceCorpusPowerReplicate(
        corpus_name=corpus,
        scenario=scenario,
        generator_source_eta=float(source_eta),
        selected=tuple(BLOCK_CANDIDATE for _ in range(5)),
        selector_rejected=truth,
        maximum_inner_gain=gain,
        endpoint_component_gains=np.full((components, 2), gain),
        prompt_block_ids=np.arange(components) // 10,
        topology_component_advantage=(
            np.full(components, gain) if truth else None
        ),
        call_loading=0.0,
        persistent_loading=0.0,
        request_loading=0.0,
    )


def fake_null_worker(task):
    _phase, _k, _g, _r, null_type, corpus, source_eta, start, stop = task
    return (
        null_type, corpus, float(source_eta), int(start),
        tuple(float(draw) / 100.0 for draw in range(start, stop)),
    )


def fake_power_corpus_worker(task):
    (
        _phase, _k, components, _r, scenario, corpus, source_eta,
        start, stop, _threshold,
    ) = task
    return scenario, corpus, float(source_eta), int(start), tuple(
        fake_corpus_power_record(corpus, scenario, source_eta, components)
        for _replicate in range(start, stop)
    )


def fake_combine_power(records_by_corpus, _designs, scenario, **_kwargs):
    return fake_power_record(
        scenario.name,
        records_by_corpus["exploratory"].generator_source_eta,
        records_by_corpus["fresh"].generator_source_eta,
    )


def fake_prepare_multiplier(endpoint_gains, _prompt_ids, _design, **kwargs):
    return {
        "corpus_name": kwargs["corpus_name"],
        "endpoint_identity": id(endpoint_gains),
        "multiplier_seed": kwargs["multiplier_seed"],
    }


def test_exact_and_custom_configuration_contracts_are_distinct():
    assert runner.parse_configuration("64:160:3") == (64, 160, 3)
    with pytest.raises(Exception):
        runner.parse_configuration("64:160")
    resolved = exact_resolved()
    assert resolved["configurations"] == runner.FULL_CONFIGURATIONS
    assert resolved["source_eta_grid"] == runner.SOURCE_ETA_GRID
    assert resolved["item_rhos"] == runner.DEFAULT_RHOS
    assert resolved["null_types"] == runner.FULL_NULL_TYPES
    assert resolved["null_draws"] == 1999
    assert resolved["power_replicates"] == 200
    assert resolved["confirmation_replicates"] == 200
    assert resolved["multiplier_draws"] == 999
    assert len(resolved["configurations"]) == 18
    assert {
        configuration
        for configuration in resolved["configurations"]
        if configuration[2] == 4
    } == {
        (region_count, components, 4)
        for region_count in runner.FULL_REGION_COUNTS
        for components in runner.FULL_REPEAT4_COMPONENT_COUNTS
    }
    assert len(runner._source_eta_pairs(runner.SOURCE_ETA_GRID)) == 25
    assert runner._is_exact_full_configuration(resolved)
    custom = tiny_resolved()
    assert custom["mode"] == "custom-diagnostic"
    assert not runner._is_exact_full_configuration(custom)


@pytest.mark.parametrize(
    "override",
    [
        ["--config", "64:160:3"],
        ["--source-eta-grid", "0", ".025", ".05", ".1", ".2"],
        ["--null-types", "block_null", "source_smooth_mean_null"],
        ["--scenarios", "block_null"],
        ["--null-draws", "1999"],
        ["--power-replicates", "200"],
        ["--confirmation-replicates", "200"],
        ["--seed", str(runner.FULL_SEED)],
    ],
)
def test_exact_mode_rejects_scientific_overrides_even_when_values_match(override):
    args = runner.build_arg_parser().parse_args(["--full-prereg", *override])
    with pytest.raises(ValueError, match="rejects scientific overrides"):
        runner.resolve_configuration(args)


def test_source_bundle_is_exact_and_provenance_is_path_free():
    bundle = load_bundle()
    identity = runner.source_design_bundle_identity(bundle)
    assert identity == {
        "size_bytes": 969_914,
        "sha256": "da7c2ec6d003150aeb0465eb099508aea9918b495ff00ae25ea3f6e44cfe5fb9",
    }
    assert bundle["parent"]["full_payload_record"] == {
        "size_bytes": 2_767_735,
        "sha256": "bf9a09c35e54bd36c2e7efea19c432ccf1e9105ff67c4154cfc1c6e744a843b2",
    }
    designs = runner.build_designs(bundle, 64, 160)
    assert runner._validate_bundle_source_eta_grid(bundle) == runner.SOURCE_ETA_GRID
    assert tuple(designs) == runner.REQUIRED_SOURCE_CORPORA
    assert all(design.component_count == 160 for design in designs.values())
    provenance = runner._provenance(bundle)
    assert provenance["source_design_bundle"] == identity
    encoded = json.dumps(provenance, sort_keys=True)
    assert str(runner.DEFAULT_SOURCE_DESIGN) not in encoded
    assert "filepath" not in encoded


def test_bundle_source_eta_grid_mismatch_fails_at_startup():
    bundle = load_bundle()
    changed = copy.deepcopy(bundle)
    changed["configuration"]["source_eta_grid"] = [0.0]
    with pytest.raises(ValueError, match="does not equal the scientific grid"):
        runner.build_scientific_payload(tiny_resolved(), changed)


def test_finite_rank_and_exact_clopper_pearson_boundaries():
    threshold, rank = runner.finite_null_threshold(np.arange(1999.0))
    assert rank == 1900
    assert threshold == 1899.0
    assert runner.clopper_pearson_lower(170, 200) >= 0.80
    assert runner.clopper_pearson_lower(169, 200) < 0.80
    assert runner.clopper_pearson_upper(0, 200) < 0.02
    family_size = runner.confirmation_binomial_family_size()
    assert family_size == 425
    alpha = 0.05 / family_size
    assert runner.clopper_pearson_lower(180, 200, alpha) >= 0.80
    assert runner.clopper_pearson_lower(179, 200, alpha) < 0.80
    assert runner.clopper_pearson_upper(5, 200, alpha) <= 0.10
    assert runner.clopper_pearson_upper(6, 200, alpha) > 0.10


def test_discovery_and_confirmation_decisions_use_joint_exact_bounds():
    resolved = exact_resolved()
    discovery = runner.evaluate_configuration(
        complete_fake_rows(), resolved,
        phase="discovery", repeats=3, null_complete=True,
    )
    assert discovery["evaluable"] is True
    assert discovery["pass"] is True
    confirmation = runner.evaluate_configuration(
        complete_fake_rows(), resolved,
        phase="confirmation", repeats=3, null_complete=True,
    )
    assert confirmation["pass"] is True
    assert confirmation["binomial_family_size"] == 425
    weak = runner.evaluate_configuration(
        complete_fake_rows(events=169), resolved,
        phase="discovery", repeats=3, null_complete=True,
    )
    assert weak["pass"] is False


def test_pair_grid_is_mandatory_but_nonidentification_is_not_a_separate_gate():
    resolved = exact_resolved()
    diagonal = [
        row for row in complete_fake_rows()
        if len(set(row["generator_source_eta_by_corpus"].values())) == 1
    ]
    incomplete = runner.evaluate_configuration(
        diagonal, resolved, phase="discovery", repeats=3, null_complete=True
    )
    assert incomplete["evaluable"] is False and incomplete["pass"] is False

    rows = complete_fake_rows()
    for row in rows:
        if row["scenario"] in ("block_null", "mean_only"):
            row["all_replicates_inference_identified"] = False
            row["inference_nonidentified_replicates"] = row["replicates"]
    decision = runner.evaluate_configuration(
        rows, resolved, phase="discovery", repeats=3, null_complete=True
    )
    assert decision["pass"] is True
    assert decision[
        "all_scenario_replicates_inference_identified_diagnostic"
    ] is False

    impossible = complete_fake_rows()
    impossible_primary = next(
        row for row in impossible
        if row["scenario"] in runner.PRIMARY_TRUTH_SCENARIOS
    )
    impossible_primary["all_replicates_inference_identified"] = True
    impossible_primary["inference_nonidentified_replicates"] = 1
    with pytest.raises(ValueError, match="internally inconsistent"):
        runner.evaluate_configuration(
            impossible, resolved,
            phase="discovery", repeats=3, null_complete=True,
        )


def test_incomplete_custom_and_repeat4_decisions_always_fail_closed():
    custom = runner.evaluate_configuration(
        [fake_aggregate_row("block_null", 0.0, 0.0)], tiny_resolved(),
        phase="discovery", repeats=3, null_complete=True,
    )
    assert custom["evaluable"] is False and custom["pass"] is False
    repeat4 = runner.evaluate_configuration(
        complete_fake_rows(), exact_resolved(),
        phase="discovery", repeats=4, null_complete=True,
    )
    assert repeat4["evaluable"] is False and repeat4["pass"] is False


def _fake_configuration_result(configuration, phase, *, confirmation_pass=True):
    k, g, r = configuration
    discovery_pass = r == 3 and g >= 320 and k >= 96
    passed = confirmation_pass if phase == "confirmation" else discovery_pass
    return {
        "phase": phase,
        "region_count": k,
        "components_per_required_corpus": g,
        "repeats": r,
        "source_design_identity": {},
        "null_calibration": {"complete": True},
        "scenarios": [],
        "decision": {"evaluable": r == 3, "pass": passed},
    }


def test_smallest_G_then_coarsest_K_gets_fresh_fixed_confirmation(monkeypatch):
    bundle = load_bundle()

    def fake_run(configuration, _resolved, _bundle, *, phase, **_kwargs):
        return _fake_configuration_result(configuration, phase)

    monkeypatch.setattr(runner, "run_configuration", fake_run)
    payload = runner.build_scientific_payload(exact_resolved(), bundle)
    assert payload["selection"]["provisional_design"] == {
        "region_count": 96,
        "components_per_required_corpus": 320,
        "repeats": 3,
    }
    assert payload["confirmation_result"]["phase"] == "confirmation"
    assert payload["authorization"][
        "attempted_input_identity_inventory_unlocked"
    ] is True
    assert sum(payload["authorization"].values()) == 1


def test_failed_confirmation_does_not_fall_through_to_another_design(monkeypatch):
    bundle = load_bundle()

    def fake_run(configuration, _resolved, _bundle, *, phase, **_kwargs):
        return _fake_configuration_result(
            configuration, phase, confirmation_pass=False
        )

    monkeypatch.setattr(runner, "run_configuration", fake_run)
    payload = runner.build_scientific_payload(exact_resolved(), bundle)
    assert payload["selection"]["provisional_design"]["region_count"] == 96
    assert payload["selection"]["confirmation_pass"] is False
    assert not any(payload["authorization"].values())


def test_custom_mode_cannot_select_or_confirm_even_if_worker_claims_pass(monkeypatch):
    bundle = load_bundle()

    def fake_run(configuration, _resolved, _bundle, *, phase, **_kwargs):
        return _fake_configuration_result(configuration, phase)

    monkeypatch.setattr(runner, "run_configuration", fake_run)
    payload = runner.build_scientific_payload(tiny_resolved(), bundle)
    assert payload["selection"]["provisional_design"] is None
    assert payload["confirmation_result"] is None
    assert not any(payload["authorization"].values())


@contextmanager
def thread_process_pool(workers, *_args, **_kwargs):
    if workers == 1:
        yield None
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            yield executor


def test_tiny_monkeypatched_serial_parallel_and_checkpoint_resume(
    tmp_path, monkeypatch
):
    bundle = load_bundle()
    resolved = tiny_resolved()
    provenance = runner._provenance(bundle)
    fingerprint = runner._run_fingerprint(resolved, provenance)
    monkeypatch.setattr(runner, "_null_worker", fake_null_worker)
    monkeypatch.setattr(runner, "_power_corpus_worker", fake_power_corpus_worker)
    monkeypatch.setattr(
        runner, "combine_source_power_corpus_replicates", fake_combine_power
    )
    monkeypatch.setattr(
        runner, "prepare_graph_aware_source_corpus_multiplier",
        fake_prepare_multiplier,
    )
    monkeypatch.setattr(runner, "_process_pool", thread_process_pool)
    serial = runner.run_configuration(
        (64, 160, 3), resolved, bundle, workers=1
    )
    for corpus in runner.REQUIRED_SOURCE_CORPORA:
        diagnostics = serial["source_design_identity"][corpus][
            "source_split_diagnostics"
        ]
        assert diagnostics["gates"]["all_source_split_gates_pass"] is True
        prompt_source = diagnostics["prompt_by_source"]
        assert len(prompt_source["incidence_counts"]) == prompt_source[
            "prompt_blocks"
        ]
        assert sum(map(sum, prompt_source["incidence_counts"])) == 160
    parallel = runner.run_configuration(
        (64, 160, 3), resolved, bundle, workers=2
    )
    assert runner.serialize_payload(serial) == runner.serialize_payload(parallel)

    checkpoint = runner.CheckpointStore(
        tmp_path / "checkpoint", resolved, provenance, fingerprint
    )
    first = runner.run_configuration(
        (64, 160, 3), resolved, bundle,
        workers=2, checkpoint_store=checkpoint,
    )

    def unexpected(_task):
        raise AssertionError("complete checkpoints must not recompute")

    monkeypatch.setattr(runner, "_null_worker", unexpected)
    monkeypatch.setattr(runner, "_power_corpus_worker", unexpected)
    resumed = runner.run_configuration(
        (64, 160, 3), resolved, bundle,
        workers=2, checkpoint_store=checkpoint,
    )
    assert runner.serialize_payload(first) == runner.serialize_payload(resumed)
    assert not list((tmp_path / "checkpoint").rglob("*.tmp"))
    power_shards = list((tmp_path / "checkpoint").rglob("power_chunks/**/*.json"))
    assert len(power_shards) == 1
    shard = runner._read_checkpoint(power_shards[0])
    assert "pair_records" in shard
    assert "corpus_records" not in shard


def test_two_corpus_eta_grid_forms_every_pair_and_reuses_common_random_streams(
    monkeypatch,
):
    bundle = load_bundle()
    resolved = copy.deepcopy(tiny_resolved())
    resolved["source_eta_grid"] = (0.0, 0.025)
    observed_multiplier_seeds = {}
    prepared_calls = []

    def observing_combine(records, designs, scenario, **kwargs):
        pair = tuple(
            records[corpus].generator_source_eta
            for corpus in runner.REQUIRED_SOURCE_CORPORA
        )
        observed_multiplier_seeds.setdefault(kwargs["multiplier_seed"], set()).add(pair)
        assert set(kwargs["prepared_multiplier_components_by_corpus"]) == set(
            runner.REQUIRED_SOURCE_CORPORA
        )
        assert (
            records["exploratory"].call_loading
            == records["fresh"].call_loading
        )
        return fake_combine_power(records, designs, scenario, **kwargs)

    def observing_prepare(*args, **kwargs):
        prepared_calls.append((kwargs["corpus_name"], kwargs["multiplier_seed"]))
        return fake_prepare_multiplier(*args, **kwargs)

    def indexed_power_worker(task):
        (
            _phase, _k, components, _r, scenario, corpus, source_eta,
            start, stop, _threshold,
        ) = task
        return scenario, corpus, float(source_eta), int(start), tuple(
            replace(
                fake_corpus_power_record(
                    corpus, scenario, source_eta, components
                ),
                call_loading=float(replicate),
            )
            for replicate in range(start, stop)
        )

    monkeypatch.setattr(runner, "_null_worker", fake_null_worker)
    monkeypatch.setattr(runner, "_power_corpus_worker", indexed_power_worker)
    monkeypatch.setattr(
        runner, "combine_source_power_corpus_replicates", observing_combine
    )
    monkeypatch.setattr(
        runner, "prepare_graph_aware_source_corpus_multiplier",
        observing_prepare,
    )
    result = runner.run_configuration((64, 160, 3), resolved, bundle)
    assert len(result["null_calibration"]["cells"]) == 4
    assert len(result["scenarios"]) == 4
    assert all(len(pairs) == 4 for pairs in observed_multiplier_seeds.values())
    assert len(observed_multiplier_seeds) == resolved["power_replicates"]
    assert len(prepared_calls) == (
        2 * len(resolved["source_eta_grid"]) * resolved["power_replicates"]
    )

    common = (resolved, "discovery", 64, 160, 3)
    assert runner._null_corpus_seed(
        *common, "block_null", "exploratory", 0.0, 7
    ) == runner._null_corpus_seed(
        *common, "block_null", "exploratory", 0.2, 7
    )
    assert runner._power_corpus_seed(
        *common, "block_null", "fresh", 0.0, 9
    ) == runner._power_corpus_seed(
        *common, "block_null", "fresh", 0.2, 9
    )
    assert runner._null_base_seed(
        resolved, "discovery", 64, 160, 3, "block_null", 7
    ) != runner._null_base_seed(
        resolved, "confirmation", 64, 160, 3, "block_null", 7
    )
    assert runner._power_multiplier_seed(
        resolved, "discovery", 64, 160, 3, "block_null", 9
    ) != runner._power_multiplier_seed(
        resolved, "confirmation", 64, 160, 3, "block_null", 9
    )


def test_chunk_checkpoint_resume_recomputes_only_missing_shards(
    tmp_path, monkeypatch
):
    bundle = load_bundle()
    resolved = copy.deepcopy(tiny_resolved())
    resolved["null_draws"] = 3
    resolved["power_replicates"] = 3
    monkeypatch.setattr(runner, "NULL_DRAW_CHUNK_SIZE", 2)
    monkeypatch.setattr(runner, "POWER_REPLICATE_CHUNK_SIZE", 2)
    monkeypatch.setattr(runner, "_null_worker", fake_null_worker)
    monkeypatch.setattr(runner, "_power_corpus_worker", fake_power_corpus_worker)
    monkeypatch.setattr(
        runner, "combine_source_power_corpus_replicates", fake_combine_power
    )
    monkeypatch.setattr(
        runner, "prepare_graph_aware_source_corpus_multiplier",
        fake_prepare_multiplier,
    )
    provenance = runner._provenance(bundle)
    store = runner.CheckpointStore(
        tmp_path / "checkpoint", resolved, provenance,
        runner._run_fingerprint(resolved, provenance),
    )
    runner.run_configuration(
        (64, 160, 3), resolved, bundle, checkpoint_store=store
    )
    null_shards = sorted((tmp_path / "checkpoint").rglob("null_chunks/**/*.json"))
    power_shards = sorted((tmp_path / "checkpoint").rglob("power_chunks/**/*.json"))
    assert len(null_shards) == 2 and len(power_shards) == 2
    (tmp_path / "checkpoint" / "discovery" / "K64_G160_R3" / "null_barrier.json").unlink()
    null_shards[-1].unlink()
    power_shards[-1].unlink()
    calls = {"null": 0, "power": 0}

    def counting_null(task):
        calls["null"] += 1
        return fake_null_worker(task)

    def counting_power(task):
        calls["power"] += 1
        return fake_power_corpus_worker(task)

    monkeypatch.setattr(runner, "_null_worker", counting_null)
    monkeypatch.setattr(runner, "_power_corpus_worker", counting_power)
    runner.run_configuration(
        (64, 160, 3), resolved, bundle, checkpoint_store=store
    )
    assert calls == {"null": 2, "power": 2}


def test_real_spawn_executor_is_order_independent_and_requests_one_thread():
    before = {name: os.environ.get(name) for name in runner._BLAS_ENVIRONMENT_VARIABLES}
    with runner._single_thread_child_environment():
        assert all(os.environ[name] == "1" for name in runner._BLAS_ENVIRONMENT_VARIABLES)
    assert {name: os.environ.get(name) for name in runner._BLAS_ENVIRONMENT_VARIABLES} == before
    with runner._process_pool(2) as executor:
        observed = sorted(runner._execute_tasks([-3, 2, -1], abs, executor))
    assert observed == [1, 2, 3]


def test_parallel_task_submission_is_bounded():
    class TrackingExecutor:
        def __init__(self):
            self._max_workers = 2
            self.inner = ThreadPoolExecutor(max_workers=self._max_workers)
            self.submitted = 0
            self.completed = 0
            self.maximum_outstanding = 0

        def submit(self, function, value):
            self.submitted += 1
            self.maximum_outstanding = max(
                self.maximum_outstanding, self.submitted - self.completed
            )
            future = self.inner.submit(function, value)
            future.add_done_callback(lambda _future: setattr(
                self, "completed", self.completed + 1
            ))
            return future

    executor = TrackingExecutor()
    try:
        assert sorted(runner._execute_tasks(range(50), abs, executor)) == list(range(50))
    finally:
        executor.inner.shutdown()
    assert executor.maximum_outstanding <= 4


def test_checkpoint_integrity_schema_and_conflicting_overwrite_fail_closed(tmp_path):
    path = tmp_path / "unit.json"
    runner._write_checkpoint(path, {"kind": "unit", "value": 1})
    runner._write_checkpoint(path, {"kind": "unit", "value": 1})
    with pytest.raises(ValueError, match="conflicting checkpoint overwrite"):
        runner._write_checkpoint(path, {"kind": "unit", "value": 2})
    envelope = json.loads(path.read_text())
    envelope["payload"]["value"] = 3
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="integrity hash mismatch"):
        runner._read_checkpoint(path)


def test_checkpoint_store_reopen_rejects_changed_config_seed_and_provenance(
    tmp_path,
):
    bundle = load_bundle()
    resolved = tiny_resolved()
    provenance = runner._provenance(bundle)
    fingerprint = runner._run_fingerprint(resolved, provenance)
    checkpoint_root = tmp_path / "checkpoint"

    runner.CheckpointStore(checkpoint_root, resolved, provenance, fingerprint)
    reopened = runner.CheckpointStore(
        checkpoint_root, resolved, provenance, fingerprint
    )
    assert reopened.fingerprint == fingerprint

    changed_configuration = copy.deepcopy(resolved)
    changed_configuration["null_draws"] += 1
    with pytest.raises(
        ValueError, match="checkpoint fingerprint/configuration mismatch"
    ):
        runner.CheckpointStore(
            checkpoint_root,
            changed_configuration,
            provenance,
            runner._run_fingerprint(changed_configuration, provenance),
        )

    changed_seed = copy.deepcopy(resolved)
    changed_seed["seed"] += 1
    with pytest.raises(
        ValueError, match="checkpoint fingerprint/configuration mismatch"
    ):
        runner.CheckpointStore(
            checkpoint_root,
            changed_seed,
            provenance,
            runner._run_fingerprint(changed_seed, provenance),
        )

    changed_provenance = copy.deepcopy(provenance)
    changed_provenance["files"]["source_science"]["sha256"] = "0" * 64
    with pytest.raises(
        ValueError, match="checkpoint fingerprint/configuration mismatch"
    ):
        runner.CheckpointStore(
            checkpoint_root,
            resolved,
            changed_provenance,
            runner._run_fingerprint(resolved, changed_provenance),
        )


def test_power_checkpoint_record_roundtrip_and_extra_key_rejection():
    record = fake_power_record("cumulative_rho_0.10", 0.025, 0.10)
    encoded = runner._power_record_to_json(record)
    decoded = runner._power_record_from_json(encoded)
    assert runner._power_record_to_json(decoded) == encoded
    assert encoded["generator_source_eta_by_corpus"] == {
        "exploratory": 0.025,
        "fresh": 0.10,
    }
    assert encoded["inference_source_eta_grid"] == [0.0]
    assert decoded.multiplier_order_statistic_rank_one_based == 3
    assert decoded.inference_identified is True
    assert decoded.inference_prompt_blocks == (16, 16)
    assert decoded.inference_source_regions == (40, 64)
    assert all(
        set(candidate) == {"gamma", "rho_item"}
        for corpus in encoded["selected"]
        for candidate in corpus
    )
    bad = copy.deepcopy(encoded)
    bad["unexpected"] = True
    with pytest.raises(ValueError, match="invalid schema"):
        runner._power_record_from_json(bad)


def test_summary_projection_binds_full_payload_and_omits_scenario_bulk(monkeypatch):
    bundle = load_bundle()

    def fake_run(configuration, _resolved, _bundle, *, phase, **_kwargs):
        row = _fake_configuration_result(configuration, phase)
        row["scenarios"] = [fake_aggregate_row("block_null", 0.0, 0.0)]
        return row

    monkeypatch.setattr(runner, "run_configuration", fake_run)
    payload = runner.build_scientific_payload(tiny_resolved(), bundle)
    projection = runner.project_summary(payload)
    expected = runner._content_record(runner.serialize_payload(payload).encode())
    assert projection["full_payload_record"] == expected
    assert "scenarios" not in projection["discovery_results"][0]
    assert "scenario_results_record" in projection["discovery_results"][0]
    assert "scenarios" in payload["discovery_results"][0]


def test_atomic_output_and_default_exit_contract(tmp_path, monkeypatch):
    payload = {
        "authorization": {
            "attempted_input_identity_inventory_unlocked": False,
        }
    }
    first = tmp_path / "a" / "one.json"
    second = tmp_path / "b" / "two.json"
    runner.write_payload(first, payload)
    runner.write_payload(second, payload)
    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.rglob("*.tmp"))
    monkeypatch.setattr(runner, "main", lambda _args: payload)
    assert runner.cli([]) == 2
    assert runner.cli(["--audit-only"]) == 0
    payload["authorization"]["attempted_input_identity_inventory_unlocked"] = True
    assert runner.cli([]) == 0


def test_start_end_provenance_recheck_detects_scientific_change(monkeypatch):
    bundle = load_bundle()
    snapshot = runner._provenance(bundle)
    changed = copy.deepcopy(snapshot)
    changed["files"]["runner"]["sha256"] = "0" * 64
    monkeypatch.setattr(runner, "_provenance", lambda _bundle: changed)
    with pytest.raises(RuntimeError, match="changed during the run"):
        runner._assert_provenance_unchanged(snapshot, bundle=bundle)


def test_blas_identity_drops_absolute_library_paths():
    common = {
        "user_api": "blas",
        "internal_api": "openblas",
        "prefix": "libopenblas",
        "version": "0.3.26",
        "threading_layer": "pthreads",
        "architecture": "Haswell",
    }
    first = runner._portable_blas_runtime([{**common, "filepath": "/a/lib.so"}])
    second = runner._portable_blas_runtime([{**common, "filepath": "/b/lib.so"}])
    assert first == second
    assert "filepath" not in first[0]

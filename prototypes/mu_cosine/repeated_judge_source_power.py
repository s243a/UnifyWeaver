"""Source-dependent Stage-A extensions for the repeated-judge power harness.

This module is synthetic and no-spend.  It keeps the existing within-component
candidate family and nuisance fits, while adding the frozen topology-only
source dependence as a *sampling* sensitivity.  Candidate selection and the
two primary endpoints use component-marginal quasi likelihoods.  Prompt and
source dependence enter the generator, source-atomic splits, and the
conservative simultaneous multiplier inference.

``JointPosterior`` is deliberately absent: it remains the later learned and
calibrated decision comparator, not a covariance factorization or a source
kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from repeated_judge_power import (
    BLOCK_CANDIDATE,
    CALL_CHANNEL_COVARIANCE,
    CHANNELS,
    DEFAULT_GAMMAS,
    DEFAULT_MEAN_RIDGES,
    DEFAULT_RHOS,
    INNER_FOLDS,
    MAX_PROMPT_ROWS,
    OUTER_FOLDS,
    PERSISTENT_CHANNEL_COVARIANCE,
    PRIMARY_ENDPOINTS,
    REQUEST_CHANNEL_COVARIANCE,
    ROWS_PER_COMPONENT,
    SCENARIO_BY_NAME,
    WAVE_CHANNEL_COVARIANCE,
    Candidate,
    CandidateSummary,
    CampaignGeometry,
    ComponentSplits,
    InnerSearch,
    OuterFold,
    RepeatedScenario,
    build_campaign_geometry,
    candidate_covariance,
    candidate_grid,
    component_gaussian_nll,
    deranged_item_kernel,
    derive_seed,
    draw_latent_states,
    fit_repeat_nuisance,
    gamma_item_kernel,
    posterior_state_nll,
    rho_matched_correlation,
    scenario_item_kernel,
    select_strictly_calibrated,
    summary,
)


SOURCE_ETA_GRID = (0.0, 0.025, 0.05, 0.10, 0.20)
REQUIRED_SOURCE_CORPORA = ("exploratory", "fresh")
SOURCE_SMOOTH_MEAN_STRENGTH = 0.10


class MultiplierNotIdentifiedError(ValueError):
    """Raised only when the frozen multiplier has zero/nonfinite scale."""


def _symmetric(value):
    value = np.asarray(value, dtype=float)
    return 0.5 * (value + value.T)


def _validate_source_eta(value):
    if isinstance(value, bool):
        raise ValueError("source eta must be a frozen numeric grid value")
    value = float(value)
    if value not in SOURCE_ETA_GRID:
        raise ValueError(f"source eta must belong to {SOURCE_ETA_GRID}")
    return value


def _validate_source_eta_grid(values):
    values = tuple(_validate_source_eta(value) for value in values)
    if not values or len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("source eta grid must be nonempty, unique, and increasing")
    return values


def _validate_source_eta_by_corpus(values):
    """Return generator source strengths in the frozen corpus order.

    Source dependence is a nuisance property of each corpus, not a shared
    effect-size parameter.  Requiring the explicit mapping prevents the five
    diagonal ``(eta, eta)`` worlds from being mistaken for the complete 25-pair
    nuisance grid.
    """

    if not isinstance(values, Mapping) or set(values) != set(
        REQUIRED_SOURCE_CORPORA
    ):
        raise ValueError(
            "source_eta_by_corpus must contain exactly exploratory and fresh"
        )
    return tuple(
        _validate_source_eta(values[corpus])
        for corpus in REQUIRED_SOURCE_CORPORA
    )


def _validated_optional_splits_mapping(splits_by_corpus):
    if splits_by_corpus is None:
        return (None,) * len(REQUIRED_SOURCE_CORPORA)
    if not isinstance(splits_by_corpus, Mapping) or set(splits_by_corpus) != set(
        REQUIRED_SOURCE_CORPORA
    ):
        raise ValueError(
            "splits_by_corpus must contain exactly exploratory and fresh"
        )
    output = tuple(splits_by_corpus[corpus] for corpus in REQUIRED_SOURCE_CORPORA)
    if any(not isinstance(value, ComponentSplits) for value in output):
        raise TypeError("every cached source split must be a ComponentSplits")
    return output


@dataclass(frozen=True)
class SourceDesign:
    """Exact component lift of one frozen source-region exposure matrix."""

    region_ids: tuple[str, ...]
    exposure: np.ndarray
    exposure_factor: np.ndarray
    assignment_region_ids: tuple[str, ...]
    component_region_index: np.ndarray
    incidence: np.ndarray

    @property
    def component_count(self):
        return len(self.assignment_region_ids)

    @property
    def region_count(self):
        return len(self.region_ids)

    @property
    def counts_by_region(self):
        counts = np.bincount(
            self.component_region_index, minlength=self.region_count
        )
        return {
            region: int(counts[index])
            for index, region in enumerate(self.region_ids)
        }


def build_source_design(region_ids, exposure, assignment_region_ids):
    """Validate and freeze an exact ``H E H.T`` source design.

    The registered matrices are full rank, so a Cholesky factor is used.  A
    rank-deficient or numerically repaired replacement is not silently
    accepted as the frozen Stage-A input.
    """

    region_ids = tuple(region_ids)
    if (
        not region_ids
        or region_ids != tuple(sorted(set(region_ids)))
        or any(not isinstance(region, str) or not region for region in region_ids)
    ):
        raise ValueError("region IDs must be unique nonempty strings in canonical order")
    matrix = _symmetric(exposure)
    size = len(region_ids)
    if (
        matrix.shape != (size, size)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, np.asarray(exposure), atol=1e-12, rtol=0.0)
        or not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)
        or np.min(matrix) < -1e-12
    ):
        raise ValueError("source exposure must be finite, symmetric, nonnegative, and unit diagonal")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the frozen Stage-A source exposure must be positive definite") from exc

    assignments = tuple(assignment_region_ids)
    if not assignments or any(region not in set(region_ids) for region in assignments):
        raise ValueError("every component must have a canonical source-region assignment")
    index_by_region = {region: index for index, region in enumerate(region_ids)}
    component_index = np.asarray(
        [index_by_region[region] for region in assignments], dtype=int
    )
    incidence = np.zeros((len(assignments), size), dtype=float)
    incidence[np.arange(len(assignments)), component_index] = 1.0
    for value in (matrix, factor, component_index, incidence):
        value.setflags(write=False)
    return SourceDesign(
        region_ids,
        matrix,
        factor,
        assignments,
        component_index,
        incidence,
    )


def source_component_factor(design: SourceDesign, source_eta):
    """Return ``A`` such that ``A A.T = (1-eta)I + eta H E H.T``."""

    if not isinstance(design, SourceDesign):
        raise TypeError("design must be a SourceDesign")
    eta = _validate_source_eta(source_eta)
    identity = math.sqrt(1.0 - eta) * np.eye(design.component_count)
    region = math.sqrt(eta) * design.incidence @ design.exposure_factor
    return np.concatenate((identity, region), axis=1)


def source_component_correlation(design: SourceDesign, source_eta):
    factor = source_component_factor(design, source_eta)
    return _symmetric(factor @ factor.T)


def _list_schedule_region_counts(counts, bins):
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("list-scheduling bins must be an integer of at least two")
    active = [region for region in sorted(counts) if counts[region] > 0]
    if len(active) < bins:
        raise ValueError("source-atomic scheduling needs at least one active region per bin")
    loads = [0] * bins
    group_counts = [0] * bins
    assignment = {}
    for region in sorted(active, key=lambda value: (-counts[value], value)):
        winner = min(
            range(bins),
            key=lambda index: (
                loads[index] + counts[region],
                group_counts[index] + 1,
                index,
            ),
        )
        assignment[region] = winner
        loads[winner] += counts[region]
        group_counts[winner] += 1
    return assignment, tuple(loads)


def _least_source_prompt_blocks(positions, design, max_prompt_rows):
    """Pack source-grouped components by least ``(same-source,total,block-id)``."""

    positions = tuple(sorted(map(int, positions)))
    block_count = int(math.ceil(len(positions) / max_prompt_rows))
    blocks = [[] for _ in range(block_count)]
    source_counts = [dict() for _ in range(block_count)]
    positions_by_region = {}
    for position in positions:
        region = design.assignment_region_ids[position]
        positions_by_region.setdefault(region, []).append(position)
    region_order = sorted(
        positions_by_region,
        key=lambda region: (-len(positions_by_region[region]), region),
    )
    for region in region_order:
        for position in positions_by_region[region]:
            eligible = [
                index for index, block in enumerate(blocks)
                if len(block) < max_prompt_rows
            ]
            if not eligible:
                raise AssertionError("prompt packer exhausted preallocated capacity")
            winner = min(
                eligible,
                key=lambda index: (
                    source_counts[index].get(region, 0),
                    len(blocks[index]),
                    index,
                ),
            )
            blocks[winner].append(position)
            source_counts[winner][region] = (
                source_counts[winner].get(region, 0) + 1
            )
    return tuple(np.asarray(block, dtype=int) for block in blocks)


def _prompt_source_components(design, prompt_blocks):
    """Connected components of the bipartite prompt x active-source graph."""

    active_regions = sorted({
        design.assignment_region_ids[int(position)]
        for block in prompt_blocks for position in block
    })
    source_offset = len(prompt_blocks)
    parent = list(range(len(prompt_blocks) + len(active_regions)))
    source_node = {
        region: source_offset + index
        for index, region in enumerate(active_regions)
    }

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for block_index, block in enumerate(prompt_blocks):
        for position in block:
            union(
                block_index,
                source_node[design.assignment_region_ids[int(position)]],
            )
    return len({find(index) for index in range(len(parent))}) if parent else 0


def source_split_diagnostics(design, splits):
    """Validate and report the frozen grouped-split and incidence gates."""

    assignments = np.asarray(design.assignment_region_ids, dtype=object)
    source_atomic = True
    for region in design.region_ids:
        positions = np.flatnonzero(assignments == region)
        if len(positions) and (
            len(set(splits.outer_label[positions])) != 1
            or len(set(splits.inner_label[positions])) != 1
        ):
            source_atomic = False
            break
    prompt_atomic = all(
        len(set(splits.outer_label[block])) == 1
        and len(set(splits.inner_label[block])) == 1
        for block in splits.prompt_blocks
    )
    counts = design.counts_by_region
    active = [region for region in design.region_ids if counts[region] > 0]
    largest_group = max(counts[region] for region in active)
    outer_rows = []
    for outer, fold in enumerate(splits.outer):
        held_regions = sorted(set(assignments[fold.held].tolist()))
        held_blocks = sorted(set(splits.prompt_block_index[fold.held].tolist()))
        outer_rows.append({
            "outer_fold": outer,
            "components": int(len(fold.held)),
            "active_source_regions": len(held_regions),
            "prompt_blocks": len(held_blocks),
            "maximum_source_share": float(
                max(np.sum(assignments[fold.held] == region) for region in held_regions)
                / len(fold.held)
            ),
            "passes_minimum_regions": len(held_regions) >= 5,
            "passes_minimum_prompt_blocks": len(held_blocks) >= 4,
        })
    outer_loads = [row["components"] for row in outer_rows]
    outer_imbalance_passes = max(outer_loads) - min(outer_loads) <= largest_group

    local_inner_rows = []
    local_inner_imbalance_passes = True
    for outer in range(OUTER_FOLDS):
        positions = np.flatnonzero(splits.outer_label == outer)
        loads = [int(np.sum(splits.inner_label[positions] == inner)) for inner in range(INNER_FOLDS)]
        regions = sorted(set(assignments[positions].tolist()))
        largest = max(counts[region] for region in regions)
        passed = max(loads) - min(loads) <= largest
        local_inner_imbalance_passes = local_inner_imbalance_passes and passed
        local_inner_rows.append({
            "outer_fold": outer,
            "global_inner_component_counts": loads,
            "spread": int(max(loads) - min(loads)),
            "largest_region_count": int(largest),
            "passes_grouped_list_bound": bool(passed),
        })

    inner_rows = []
    inner_imbalance_passes = True
    for outer, fold in enumerate(splits.outer):
        held_sizes = []
        for inner, (_fit, held) in enumerate(fold.inner):
            held_regions = sorted(set(assignments[held].tolist()))
            held_blocks = sorted(set(splits.prompt_block_index[held].tolist()))
            held_sizes.append(len(held))
            inner_rows.append({
                "left_out_outer_fold": outer,
                "global_inner_fold": inner,
                "components": int(len(held)),
                "active_source_regions": len(held_regions),
                "prompt_blocks": len(held_blocks),
                "maximum_source_share": float(
                    max(np.sum(assignments[held] == region) for region in held_regions)
                    / len(held)
                ),
                "passes_minimum_regions": len(held_regions) >= 5,
                "passes_minimum_prompt_blocks": len(held_blocks) >= 4,
            })
        included_outer = [index for index in range(OUTER_FOLDS) if index != outer]
        grouped_bound = sum(
            max(
                (counts[region] for region in active
                 if splits.outer_label[np.flatnonzero(assignments == region)[0]] == index),
                default=0,
            )
            for index in included_outer
        )
        inner_imbalance_passes = inner_imbalance_passes and (
            max(held_sizes) - min(held_sizes) <= grouped_bound
        )

    active_region_ids = sorted(set(assignments.tolist()))
    region_index = {region: index for index, region in enumerate(active_region_ids)}
    cross = np.zeros((len(splits.prompt_blocks), len(active_region_ids)), dtype=float)
    for block_index, block in enumerate(splits.prompt_blocks):
        for position in block:
            cross[block_index, region_index[design.assignment_region_ids[int(position)]]] += 1.0
    maximum_within_prompt_source_share = max(
        max(
            np.sum(
                assignments[block] == region
            ) / len(block)
            for region in set(assignments[block].tolist())
        )
        for block in splits.prompt_blocks
    )
    prompt_source = {
        "rank": int(np.linalg.matrix_rank(cross)),
        "components": int(_prompt_source_components(design, splits.prompt_blocks)),
        "maximum_source_share": float(maximum_within_prompt_source_share),
        "prompt_blocks": len(splits.prompt_blocks),
        "active_source_regions": len(active_region_ids),
        # Rows follow prompt_block_ids and columns follow
        # active_source_region_ids.  Recording this small integer table makes
        # the split-contained source-spreading claim independently auditable
        # without exposing any simulated outcomes.
        "prompt_block_ids": list(range(len(splits.prompt_blocks))),
        "active_source_region_ids": list(active_region_ids),
        "incidence_counts": cross.astype(int).tolist(),
        "analysis_signature_by_prompt_block": [
            {
                "prompt_block": int(block_index),
                "outer_fold": int(splits.outer_label[block[0]]),
                "global_inner_fold": int(splits.inner_label[block[0]]),
            }
            for block_index, block in enumerate(splits.prompt_blocks)
        ],
    }
    gates = {
        "source_atomic": source_atomic,
        "prompt_split_atomic": prompt_atomic,
        "outer_grouped_imbalance_within_list_bound": outer_imbalance_passes,
        "each_outer_local_inner_spread_within_largest_region": (
            local_inner_imbalance_passes
        ),
        "inner_grouped_imbalance_within_list_bound": inner_imbalance_passes,
        "every_outer_held_has_at_least_5_sources": all(
            row["passes_minimum_regions"] for row in outer_rows
        ),
        "every_outer_held_has_at_least_4_prompt_blocks": all(
            row["passes_minimum_prompt_blocks"] for row in outer_rows
        ),
        "every_leave_outer_inner_held_has_at_least_5_sources": all(
            row["passes_minimum_regions"] for row in inner_rows
        ),
        "every_leave_outer_inner_held_has_at_least_4_prompt_blocks": all(
            row["passes_minimum_prompt_blocks"] for row in inner_rows
        ),
    }
    gates["all_source_split_gates_pass"] = all(gates.values())
    return {
        "outer_held": outer_rows,
        "outer_local_inner_assignment": local_inner_rows,
        "leave_one_outer_inner_held": inner_rows,
        "outer_component_spread": int(max(outer_loads) - min(outer_loads)),
        "largest_source_group": int(largest_group),
        "prompt_by_source": prompt_source,
        "gates": gates,
    }


def source_atomic_component_splits(
    design: SourceDesign,
    *,
    seed=51001,
    outer_folds=OUTER_FOLDS,
    inner_folds=INNER_FOLDS,
    max_prompt_rows=MAX_PROMPT_ROWS,
):
    """Make source-atomic folds and mixed-source round-robin prompt blocks."""

    if not isinstance(design, SourceDesign):
        raise TypeError("design must be a SourceDesign")
    if outer_folds != OUTER_FOLDS or inner_folds != INNER_FOLDS:
        raise ValueError("Stage A requires exactly five outer and three global-inner folds")
    if not 1 <= int(max_prompt_rows) <= MAX_PROMPT_ROWS:
        raise ValueError("prompt capacity must lie in [1,10]")
    counts = design.counts_by_region
    outer_by_region, _outer_loads = _list_schedule_region_counts(
        counts, outer_folds
    )
    # The stable global-inner label is assigned separately within each outer
    # source bin.  This prevents a global list schedule from leaving a
    # leave-one-outer inner cell with too few dependency groups.
    inner_by_region = {}
    for outer in range(outer_folds):
        local_counts = {
            region: counts[region]
            for region in design.region_ids
            if counts[region] > 0 and outer_by_region[region] == outer
        }
        local_assignment, _local_loads = _list_schedule_region_counts(
            local_counts, inner_folds
        )
        inner_by_region.update(local_assignment)
    outer_label = np.asarray(
        [outer_by_region[region] for region in design.assignment_region_ids], dtype=int
    )
    inner_label = np.asarray(
        [inner_by_region[region] for region in design.assignment_region_ids], dtype=int
    )

    prompt_blocks = []
    for outer in range(outer_folds):
        for inner in range(inner_folds):
            positions = np.flatnonzero(
                (outer_label == outer) & (inner_label == inner)
            )
            if not len(positions):
                continue
            prompt_blocks.extend(
                _least_source_prompt_blocks(
                    positions, design, int(max_prompt_rows)
                )
            )
    if not prompt_blocks:
        raise ValueError("source scheduling produced no prompt blocks")
    flattened = np.concatenate(prompt_blocks)
    if sorted(flattened.tolist()) != list(range(design.component_count)):
        raise AssertionError("prompt blocks must partition every component exactly once")
    prompt_block_index = np.empty(design.component_count, dtype=int)
    for block_index, block in enumerate(prompt_blocks):
        if len(block) > max_prompt_rows:
            raise AssertionError("prompt block exceeded its frozen capacity")
        if len(set(outer_label[block])) != 1 or len(set(inner_label[block])) != 1:
            raise AssertionError("a prompt block crossed an analysis signature")
        prompt_block_index[block] = block_index

    all_components = set(range(design.component_count))
    outer_records = []
    for outer in range(outer_folds):
        held = np.flatnonzero(outer_label == outer)
        train = np.asarray(sorted(all_components - set(map(int, held))), dtype=int)
        if not len(held) or not len(train):
            raise ValueError("every source-atomic outer fold must have train and held components")
        inner_records = []
        train_set = set(map(int, train))
        for inner in range(inner_folds):
            inner_held = np.flatnonzero(
                (outer_label != outer) & (inner_label == inner)
            )
            inner_fit = np.asarray(
                sorted(train_set - set(map(int, inner_held))), dtype=int
            )
            if not len(inner_fit) or not len(inner_held):
                raise ValueError("every nested source-atomic split must have fit and held components")
            inner_records.append((inner_fit, inner_held))
        outer_records.append(OuterFold(train, held, tuple(inner_records)))

    for value in (outer_label, inner_label, prompt_block_index):
        value.setflags(write=False)
    result = ComponentSplits(
        tuple(outer_records),
        int(seed),
        outer_label,
        inner_label,
        tuple(prompt_blocks),
        prompt_block_index,
    )
    diagnostics = source_split_diagnostics(design, result)
    if not diagnostics["gates"]["all_source_split_gates_pass"]:
        failed = sorted(
            name for name, passed in diagnostics["gates"].items()
            if name != "all_source_split_gates_pass" and not passed
        )
        raise ValueError(
            "source split design gates failed: " + ", ".join(failed)
        )
    return result


def build_source_campaign_geometry(design, *, seed=24051, splits=None):
    if splits is None:
        splits = source_atomic_component_splits(design, seed=derive_seed(seed, "splits"))
    geometry = build_campaign_geometry(
        design.component_count,
        seed=derive_seed(seed, "geometry"),
        prompt_blocks=splits.prompt_blocks,
    )
    return geometry, splits


def source_smooth_omitted_mean(
    design: SourceDesign,
    *,
    seed,
    strength=SOURCE_SMOOTH_MEAN_STRENGTH,
):
    """Normalized E-correlated source-smooth mean omitted from the mean fit.

    The field is random across synthetic worlds but deterministic conditional
    on its independent seed.  Centering and realized-RMS normalization freeze
    the confound strength at ``.10`` rather than treating this control as an
    additional Gaussian covariance draw.
    """

    region_white = np.random.default_rng(seed).standard_normal(
        design.region_count
    )
    region_field = design.exposure_factor @ region_white
    component = region_field[design.component_region_index]
    component = component - float(np.mean(component))
    scale = float(np.sqrt(np.mean(component * component)))
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("source exposure cannot identify a nonconstant omitted mean")
    component /= scale
    row_loading = np.asarray([1.0, 0.65, -0.35])
    channel_loading = np.asarray([0.70, -0.45, 0.35, -0.25])
    return (
        float(strength)
        * component[:, None, None]
        * row_loading[None, :, None]
        * channel_loading[None, None, :]
    )


def draw_source_persistent(
    geometry: CampaignGeometry,
    scenario: RepeatedScenario,
    design: SourceDesign,
    source_eta,
    rng,
):
    """Draw the exact separable source x item x channel persistent field."""

    if geometry.component_count != design.component_count:
        raise ValueError("geometry and source design component counts disagree")
    eta = _validate_source_eta(source_eta)
    item = rho_matched_correlation(
        scenario_item_kernel(geometry, scenario), scenario.truth_rho
    )
    within_factor = np.linalg.cholesky(
        np.kron(item, PERSISTENT_CHANNEL_COVARIANCE)
    )
    independent = (
        rng.standard_normal((design.component_count, ROWS_PER_COMPONENT * CHANNELS))
        @ within_factor.T
    )
    region_white = (
        rng.standard_normal((design.region_count, ROWS_PER_COMPONENT * CHANNELS))
        @ within_factor.T
    )
    region_correlated = design.exposure_factor @ region_white
    persistent = (
        math.sqrt(1.0 - eta) * independent
        + math.sqrt(eta) * region_correlated[design.component_region_index]
    )
    return persistent.reshape(
        design.component_count, ROWS_PER_COMPONENT, CHANNELS
    )


def draw_source_repeated_field(
    geometry,
    scenario,
    design,
    source_eta,
    repeats,
    seed,
    *,
    missing_rate=0.02,
    include_wave_effect=True,
):
    """Draw ``[component,row,repeat,channel]`` under source dependence."""

    if repeats < 3:
        raise ValueError("the repeated-judge design requires at least three calls")
    if not 0.0 <= missing_rate < 0.25:
        raise ValueError("missing_rate must lie in [0,.25)")
    persistent = draw_source_persistent(
        geometry,
        scenario,
        design,
        source_eta,
        np.random.default_rng(derive_seed(seed, "persistent")),
    )
    call = np.random.default_rng(derive_seed(seed, "call")).standard_normal((
        geometry.component_count, ROWS_PER_COMPONENT, repeats, CHANNELS
    )) @ np.linalg.cholesky(CALL_CHANNEL_COVARIANCE).T
    request = np.random.default_rng(derive_seed(seed, "request")).standard_normal((
        len(geometry.prompt_blocks), ROWS_PER_COMPONENT, repeats, CHANNELS
    )) @ np.linalg.cholesky(REQUEST_CHANNEL_COVARIANCE).T
    wave = np.zeros((repeats, CHANNELS))
    if include_wave_effect:
        wave = np.random.default_rng(derive_seed(seed, "wave")).standard_normal(
            (repeats, CHANNELS)
        ) @ np.linalg.cholesky(
            WAVE_CHANNEL_COVARIANCE
        ).T
        wave -= wave.mean(axis=0, keepdims=True)
    mean = scenario.mean_strength * geometry.true_mean
    if scenario.name == "mean_only":
        mean = mean + source_smooth_omitted_mean(
            design,
            seed=derive_seed(seed, "source-smooth-mean"),
        )
    field = (
        mean[:, :, None, :]
        + persistent[:, :, None, :]
        + call
        + request[geometry.prompt_block_index]
        + wave[None, None, :, :]
    )
    if missing_rate:
        observed_request = np.random.default_rng(
            derive_seed(seed, "missingness")
        ).random((
            len(geometry.prompt_blocks), ROWS_PER_COMPONENT, repeats, 2
        )) >= missing_rate
        for block in range(len(geometry.prompt_blocks)):
            for row in range(ROWS_PER_COMPONENT):
                for family in range(2):
                    if np.sum(observed_request[block, row, :, family]) < 2:
                        observed_request[block, row, :2, family] = True
        observed = observed_request[geometry.prompt_block_index]
        field = field.copy()
        for family, start in enumerate((0, 2)):
            field[..., start:start + 2] = np.where(
                observed[..., family, None],
                field[..., start:start + 2],
                np.nan,
            )
    return field


def component_marginal_quasi_nll(residuals, nuisance, item_kernel, rho):
    covariance = candidate_covariance(nuisance, item_kernel, rho)
    return component_gaussian_nll(residuals, covariance)


def component_marginal_posterior_nll(
    residuals, nuisance, item_kernel, rho, states
):
    covariance = candidate_covariance(nuisance, item_kernel, rho)
    return posterior_state_nll(residuals, covariance, states)


def source_inner_candidate_search(
    field,
    geometry,
    outer_fold,
    *,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
):
    """Run the frozen selector with component-marginal quasi likelihood."""

    candidates = candidate_grid(gammas, rhos)
    gains = {candidate: [] for candidate in candidates}
    block_kernel = gamma_item_kernel(0.5, geometry.kernels)
    for fit, held in outer_fold.inner:
        nuisance = fit_repeat_nuisance(
            field,
            geometry,
            fit,
            held,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        block_nll = component_marginal_quasi_nll(
            nuisance.evaluate_centered_means,
            nuisance,
            block_kernel,
            0.0,
        )
        for candidate in candidates:
            if candidate.is_block:
                candidate_nll = block_nll
            else:
                candidate_nll = component_marginal_quasi_nll(
                    nuisance.evaluate_centered_means,
                    nuisance,
                    gamma_item_kernel(candidate.gamma, geometry.kernels),
                    candidate.rho,
                )
            gains[candidate].append(float(np.mean(block_nll - candidate_nll)))
    summaries = tuple(
        CandidateSummary(
            candidate,
            float(np.mean(values)),
            int(np.sum(np.asarray(values) > 0.0)),
            tuple(values),
        )
        for candidate, values in gains.items()
    )
    eligible = [
        row
        for row in summaries
        if not row.candidate.is_block
        and row.macro_gain > 0.0
        and row.positive_folds >= 2
    ]
    if not eligible:
        return InnerSearch(BLOCK_CANDIDATE, summaries, 0.0)
    best = min(
        eligible,
        key=lambda row: (
            -row.macro_gain,
            row.candidate.rho,
            abs(row.candidate.gamma - 0.5),
            row.candidate.gamma,
        ),
    )
    return InnerSearch(
        best.candidate,
        summaries,
        float(max(row.macro_gain for row in eligible)),
    )


def _prompt_positions(prompt_block_ids):
    ids = np.asarray(prompt_block_ids)
    if ids.ndim != 1:
        raise ValueError("prompt block IDs must be a vector")
    unique = sorted(set(ids.tolist()), key=repr)
    remap = {value: index for index, value in enumerate(unique)}
    return np.asarray([remap[value] for value in ids], dtype=int), len(unique)


def prompt_source_multiplier_covariance(design, prompt_block_ids, source_eta):
    """Dense reference ``C_eta + P P.T``; no intersection subtraction."""

    prompt_index, block_count = _prompt_positions(prompt_block_ids)
    if len(prompt_index) != design.component_count:
        raise ValueError("prompt IDs and source design must align")
    incidence = np.zeros((design.component_count, block_count), dtype=float)
    incidence[np.arange(design.component_count), prompt_index] = 1.0
    return _symmetric(
        source_component_correlation(design, source_eta)
        + incidence @ incidence.T
    )


def graph_aware_influence_standard_errors(
    values, prompt_block_ids, design, source_eta
):
    """Exact dense-reference SE for the frozen prompt-plus-source multiplier."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) != design.component_count:
        raise ValueError("values must be a component-aligned matrix")
    centered = values - values.mean(axis=0)
    covariance = prompt_source_multiplier_covariance(
        design, prompt_block_ids, source_eta
    )
    variance = np.einsum("ge,gh,he->e", centered, covariance, centered)
    return np.sqrt(np.maximum(variance, 0.0)) / len(values)


@dataclass(frozen=True)
class PreparedGraphAwareSourceCorpusMultiplier:
    """One corpus's reusable contribution to the joint source max-t.

    ``source_input_object_identity`` is intentionally process-local.  It lets
    the power combiner reject a prepared object that did not come from the
    exact endpoint array, prompt-ID array, and :class:`SourceDesign` currently
    being combined, without re-hashing those inputs for every one of the 25
    corpus-specific generator-source-eta pairs.
    """

    corpus_name: str
    corpus_index: int
    multiplier_seed: int
    draws: int
    inference_source_eta_grid: tuple[float, ...]
    inference_identified: bool
    point_estimates: np.ndarray
    standard_errors_by_source_eta: np.ndarray
    analytic_standard_errors_by_source_eta: np.ndarray
    standardized_negative_deviations: np.ndarray
    prompt_block_count: int
    active_source_region_count: int
    source_input_object_identity: tuple[int, int, int]


@dataclass(frozen=True)
class GraphAwareBounds:
    inference_source_eta_grid: tuple[float, ...]
    point_estimates: np.ndarray
    standard_errors_by_source_eta: np.ndarray
    analytic_standard_errors_by_source_eta: np.ndarray
    lower_bounds_by_source_eta: np.ndarray
    worst_source_eta_lower_bounds: np.ndarray
    worst_source_eta_indices: np.ndarray
    critical_value: float
    order_statistic_rank_one_based: int
    prompt_block_counts: tuple[int, ...]
    active_source_region_counts: tuple[int, ...]


def prepare_graph_aware_source_corpus_multiplier(
    endpoint_component_gains,
    prompt_block_ids,
    source_design,
    *,
    corpus_name,
    inference_source_eta_grid=SOURCE_ETA_GRID,
    draws=999,
    multiplier_seed=0,
):
    """Prepare one exact, reusable corpus contribution to the joint max-t.

    For each inference source eta, the multiplier perturbation is exactly

    ``D* = G^-1 Psi.T (xi_eta + P zeta)``.

    ``xi_eta`` uses the frozen region-factor source representation, ``P zeta``
    is the prompt-block field, and there is deliberately no intersection
    subtraction.  The RNG namespaces and corpus index are exactly those used
    by the former monolithic two-corpus implementation.  Consequently two
    prepared corpus objects can be combined without redrawing or redoing any
    component-by-draw products.
    """

    if corpus_name not in REQUIRED_SOURCE_CORPORA:
        raise ValueError("corpus_name must be exploratory or fresh")
    if not isinstance(source_design, SourceDesign):
        raise TypeError("source_design must be a SourceDesign")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("multiplier draws must be an integer of at least two")
    inference_source_eta_grid = _validate_source_eta_grid(
        inference_source_eta_grid
    )
    endpoint_count = len(PRIMARY_ENDPOINTS)
    values = np.asarray(endpoint_component_gains, dtype=float)
    if (
        values.shape != (source_design.component_count, endpoint_count)
        or not np.isfinite(values).all()
    ):
        raise ValueError("each corpus needs finite [G,2] endpoint influences")
    centered = values - values.mean(axis=0)
    prompt_index, prompt_count = _prompt_positions(prompt_block_ids)
    if len(prompt_index) != source_design.component_count or prompt_count < 2:
        raise ValueError("each corpus needs aligned IDs and at least two prompt blocks")
    active_sources = int(np.count_nonzero(np.bincount(
        source_design.component_region_index,
        minlength=source_design.region_count,
    )))
    if active_sources < 2:
        raise ValueError("each corpus needs at least two active source regions")
    prompt_sums = np.zeros((prompt_count, endpoint_count), dtype=float)
    np.add.at(prompt_sums, prompt_index, centered)
    source_sums = np.zeros(
        (source_design.region_count, endpoint_count), dtype=float
    )
    np.add.at(source_sums, source_design.component_region_index, centered)

    corpus_index = REQUIRED_SOURCE_CORPORA.index(corpus_name)
    multiplier_seed = int(multiplier_seed)
    epsilon = np.random.default_rng(
        derive_seed(multiplier_seed, "source-iid", corpus_index)
    ).standard_normal((draws, source_design.component_count))
    region_white = np.random.default_rng(
        derive_seed(multiplier_seed, "source-region", corpus_index)
    ).standard_normal((draws, source_design.region_count))
    prompt_white = np.random.default_rng(
        derive_seed(multiplier_seed, "prompt", corpus_index)
    ).choice((-1.0, 1.0), size=(draws, prompt_count))
    region_correlated = region_white @ source_design.exposure_factor.T
    prompt_deviation = (
        prompt_white @ prompt_sums / source_design.component_count
    )

    deviations = np.empty((
        draws,
        len(inference_source_eta_grid),
        endpoint_count,
    ))
    analytic_standard_errors = np.empty((
        len(inference_source_eta_grid), endpoint_count
    ))
    independent_variance = np.sum(centered * centered, axis=0)
    region_variance = np.einsum(
        "ke,kl,le->e", source_sums, source_design.exposure, source_sums
    )
    prompt_variance = np.sum(prompt_sums * prompt_sums, axis=0)
    for source_eta_index, source_eta in enumerate(inference_source_eta_grid):
        component_multiplier = (
            math.sqrt(1.0 - source_eta) * epsilon
            + math.sqrt(source_eta)
            * region_correlated[:, source_design.component_region_index]
        )
        source_deviation = (
            component_multiplier @ centered / source_design.component_count
        )
        deviations[:, source_eta_index] = source_deviation + prompt_deviation
        variance = (
            (1.0 - source_eta) * independent_variance
            + source_eta * region_variance
            + prompt_variance
        ) / (source_design.component_count**2)
        analytic_standard_errors[source_eta_index] = np.sqrt(
            np.maximum(variance, 0.0)
        )

    standard_errors = np.std(deviations, axis=0, ddof=1)
    inference_identified = bool(not (
        not np.isfinite(standard_errors).all()
        or np.any(standard_errors <= np.finfo(float).eps)
        or not np.isfinite(analytic_standard_errors).all()
        or np.any(analytic_standard_errors <= np.finfo(float).eps)
    ))
    if inference_identified:
        standardized_negative_deviations = (
            -deviations / standard_errors[None, ...]
        )
    else:
        # Preserve a cacheable, shape-complete sentinel.  The joint prepared
        # combiner raises MultiplierNotIdentifiedError before consuming these
        # zeros, and the power combiner converts that signal into its existing
        # explicit non-promotion record.  Thus one zero-scale corpus does not
        # force 25 redundant attempts to rebuild the same failed multiplier.
        standardized_negative_deviations = np.zeros_like(deviations)
    point = values.mean(axis=0)
    for value in (
        point,
        standard_errors,
        analytic_standard_errors,
        standardized_negative_deviations,
    ):
        value.setflags(write=False)
    return PreparedGraphAwareSourceCorpusMultiplier(
        corpus_name=corpus_name,
        corpus_index=corpus_index,
        multiplier_seed=multiplier_seed,
        draws=draws,
        inference_source_eta_grid=inference_source_eta_grid,
        inference_identified=inference_identified,
        point_estimates=point,
        standard_errors_by_source_eta=standard_errors,
        analytic_standard_errors_by_source_eta=analytic_standard_errors,
        standardized_negative_deviations=standardized_negative_deviations,
        prompt_block_count=prompt_count,
        active_source_region_count=active_sources,
        source_input_object_identity=(
            id(endpoint_component_gains),
            id(prompt_block_ids),
            id(source_design),
        ),
    )


def combine_prepared_graph_aware_source_corpus_multipliers(
    prepared_by_corpus,
    *,
    confidence=0.95,
):
    """Form the exact joint max-t from two prepared corpus contributions."""

    if not isinstance(prepared_by_corpus, Mapping) or set(
        prepared_by_corpus
    ) != set(REQUIRED_SOURCE_CORPORA):
        raise ValueError(
            "prepared multipliers must contain exactly exploratory and fresh"
        )
    prepared = tuple(
        prepared_by_corpus[corpus] for corpus in REQUIRED_SOURCE_CORPORA
    )
    for corpus_index, (corpus, value) in enumerate(zip(
        REQUIRED_SOURCE_CORPORA, prepared
    )):
        if not isinstance(value, PreparedGraphAwareSourceCorpusMultiplier):
            raise TypeError(
                "every prepared multiplier must be a "
                "PreparedGraphAwareSourceCorpusMultiplier"
            )
        if value.corpus_name != corpus or value.corpus_index != corpus_index:
            raise ValueError("prepared multiplier corpus identity mismatch")
    if not 0.0 < confidence < 1.0:
        raise ValueError("multiplier confidence must lie in (0,1)")
    if len({value.multiplier_seed for value in prepared}) != 1:
        raise ValueError("prepared corpora must share one multiplier seed")
    if len({value.draws for value in prepared}) != 1:
        raise ValueError("prepared corpora must share one multiplier draw count")
    if len({value.inference_source_eta_grid for value in prepared}) != 1:
        raise ValueError("prepared corpora must share one inference source eta grid")
    if any(not value.inference_identified for value in prepared):
        raise MultiplierNotIdentifiedError(
            "prompt-plus-source multiplier standard error is zero or nonfinite"
        )
    draws = prepared[0].draws
    inference_source_eta_grid = prepared[0].inference_source_eta_grid
    endpoint_count = len(PRIMARY_ENDPOINTS)
    expected_standard_error_shape = (
        len(inference_source_eta_grid), endpoint_count
    )
    expected_deviation_shape = (
        draws, len(inference_source_eta_grid), endpoint_count
    )
    for value in prepared:
        if (
            value.point_estimates.shape != (endpoint_count,)
            or value.standard_errors_by_source_eta.shape
            != expected_standard_error_shape
            or value.analytic_standard_errors_by_source_eta.shape
            != expected_standard_error_shape
            or value.standardized_negative_deviations.shape
            != expected_deviation_shape
            or not np.isfinite(value.point_estimates).all()
            or not np.isfinite(value.standard_errors_by_source_eta).all()
            or np.any(
                value.standard_errors_by_source_eta <= np.finfo(float).eps
            )
            or not np.isfinite(
                value.analytic_standard_errors_by_source_eta
            ).all()
            or np.any(
                value.analytic_standard_errors_by_source_eta
                <= np.finfo(float).eps
            )
            or not np.isfinite(value.standardized_negative_deviations).all()
        ):
            raise ValueError("prepared multiplier arrays are malformed")

    point = np.stack([value.point_estimates for value in prepared])
    standard_errors = np.stack([
        value.standard_errors_by_source_eta for value in prepared
    ])
    analytic_standard_errors = np.stack([
        value.analytic_standard_errors_by_source_eta for value in prepared
    ])
    standardized_negative_deviations = np.stack([
        value.standardized_negative_deviations for value in prepared
    ], axis=1)
    max_statistic = np.max(
        standardized_negative_deviations, axis=(1, 2, 3)
    )
    rank = min(int(math.ceil(confidence * (draws + 1))), draws)
    critical = float(np.sort(max_statistic)[rank - 1])
    lower = point[:, None, :] - critical * standard_errors
    worst = np.min(lower, axis=1)
    worst_index = np.argmin(lower, axis=1)
    for value in (
        point,
        standard_errors,
        analytic_standard_errors,
        lower,
        worst,
        worst_index,
    ):
        value.setflags(write=False)
    return GraphAwareBounds(
        inference_source_eta_grid=inference_source_eta_grid,
        point_estimates=point,
        standard_errors_by_source_eta=standard_errors,
        analytic_standard_errors_by_source_eta=analytic_standard_errors,
        lower_bounds_by_source_eta=lower,
        worst_source_eta_lower_bounds=worst,
        worst_source_eta_indices=worst_index,
        critical_value=critical,
        order_statistic_rank_one_based=rank,
        prompt_block_counts=tuple(
            value.prompt_block_count for value in prepared
        ),
        active_source_region_counts=tuple(
            value.active_source_region_count for value in prepared
        ),
    )


def graph_aware_prompt_source_lower_bounds(
    corpus_values,
    corpus_prompt_block_ids,
    corpus_source_designs,
    *,
    inference_source_eta_grid=SOURCE_ETA_GRID,
    confidence=0.95,
    draws=999,
    seed=0,
):
    """Conservative simultaneous prompt-plus-source lower bounds.

    One max-t critical value spans both endpoints, both corpora, and every
    inference source eta.  Promotion consumes the minimum lower bound over the
    inference source eta grid.  The two-stage implementation is algebraically
    and bit-for-bit identical to computing the same joint deviation tensor at
    once, while allowing corpus worlds to be reused across generator pairs.
    """

    values_by_corpus = tuple(corpus_values)
    prompt_by_corpus = tuple(corpus_prompt_block_ids)
    designs = tuple(corpus_source_designs)
    if (
        len(values_by_corpus) != len(REQUIRED_SOURCE_CORPORA)
        or len(prompt_by_corpus) != len(values_by_corpus)
        or len(designs) != len(values_by_corpus)
    ):
        raise ValueError("Stage A requires aligned exploratory and fresh inputs")
    prepared = {
        corpus: prepare_graph_aware_source_corpus_multiplier(
            values,
            prompt_ids,
            design,
            corpus_name=corpus,
            inference_source_eta_grid=inference_source_eta_grid,
            draws=draws,
            multiplier_seed=seed,
        )
        for corpus, values, prompt_ids, design in zip(
            REQUIRED_SOURCE_CORPORA,
            values_by_corpus,
            prompt_by_corpus,
            designs,
        )
    }
    return combine_prepared_graph_aware_source_corpus_multipliers(
        prepared,
        confidence=confidence,
    )


# Explicit alias matching the existing multicorpus naming convention.
multicorpus_prompt_source_lower_bounds = graph_aware_prompt_source_lower_bounds


def _validated_design_mapping(designs_by_corpus):
    if not isinstance(designs_by_corpus, Mapping) or set(designs_by_corpus) != set(
        REQUIRED_SOURCE_CORPORA
    ):
        raise ValueError("designs must contain exactly exploratory and fresh")
    designs = tuple(designs_by_corpus[corpus] for corpus in REQUIRED_SOURCE_CORPORA)
    if any(not isinstance(design, SourceDesign) for design in designs):
        raise TypeError("every corpus design must be a SourceDesign")
    if len({design.component_count for design in designs}) != 1:
        raise ValueError("both corpus designs must use the same registered G")
    if len({design.region_count for design in designs}) != 1:
        raise ValueError("both corpus designs must use the same registered K")
    return designs


def _source_replicate_context(design, seed, max_prompt_rows, splits=None):
    # Folds and prompt incidence are outcome-blind and deterministic for one
    # exact SourceDesign.  Full runners may validate/cache them once per
    # (corpus,K,G) and pass them here; geometry/mean covariates remain
    # replicate-seeded.
    if splits is None:
        splits = source_atomic_component_splits(
            design,
            seed=derive_seed(seed, "splits"),
            max_prompt_rows=max_prompt_rows,
        )
    elif not isinstance(splits, ComponentSplits):
        raise TypeError("cached source splits must be a ComponentSplits")
    geometry = build_campaign_geometry(
        design.component_count,
        derive_seed(seed, "geometry"),
        prompt_blocks=splits.prompt_blocks,
        max_prompt_rows=max_prompt_rows,
    )
    return geometry, splits


def source_corpus_null_maximum(
    design,
    scenario=SCENARIO_BY_NAME["block_null"],
    *,
    corpus_name,
    source_eta,
    repeats,
    seed,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
    splits=None,
):
    """Return one corpus's outer-fold selector maximum under a null world.

    This reusable unit lets a runner simulate each corpus/source-eta world once and
    form all 25 cross-corpus nuisance pairs without regenerating either world.
    ``seed`` is already corpus-specific; changing the other corpus's eta never
    perturbs this record's random-number stream.
    """

    if corpus_name not in REQUIRED_SOURCE_CORPORA:
        raise ValueError("corpus_name must be exploratory or fresh")
    if not isinstance(design, SourceDesign):
        raise TypeError("design must be a SourceDesign")
    eta = _validate_source_eta(source_eta)
    if scenario.truth_rho != 0.0:
        raise ValueError("source null calibration requires zero item coupling")
    geometry, splits = _source_replicate_context(
        design, seed, max_prompt_rows, splits=splits
    )
    field = draw_source_repeated_field(
        geometry,
        scenario,
        design,
        eta,
        repeats,
        derive_seed(seed, "field"),
        missing_rate=missing_rate,
    )
    searches = [
        source_inner_candidate_search(
            field,
            geometry,
            fold,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        for fold in splits.outer
    ]
    return float(max(search.maximum_eligible_gain for search in searches))


def combine_source_corpus_null_maxima(maxima_by_corpus):
    """Combine cached corpus selector maxima for one generator-source-eta pair."""

    if not isinstance(maxima_by_corpus, Mapping) or set(maxima_by_corpus) != set(
        REQUIRED_SOURCE_CORPORA
    ):
        raise ValueError(
            "maxima_by_corpus must contain exactly exploratory and fresh"
        )
    values = tuple(
        float(maxima_by_corpus[corpus]) for corpus in REQUIRED_SOURCE_CORPORA
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("each corpus null maximum must be finite and nonnegative")
    return float(max(values))


def source_null_maximum(
    designs_by_corpus,
    scenario=SCENARIO_BY_NAME["block_null"],
    *,
    source_eta_by_corpus,
    repeats,
    seed,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
    splits_by_corpus=None,
):
    """Return one joint two-corpus null/control selector maximum."""

    designs = _validated_design_mapping(designs_by_corpus)
    etas = _validate_source_eta_by_corpus(source_eta_by_corpus)
    splits = _validated_optional_splits_mapping(splits_by_corpus)
    maxima = {}
    for corpus, design, eta, cached_splits in zip(
        REQUIRED_SOURCE_CORPORA, designs, etas, splits
    ):
        corpus_seed = derive_seed(seed, corpus)
        maxima[corpus] = source_corpus_null_maximum(
            design,
            scenario,
            corpus_name=corpus,
            source_eta=eta,
            repeats=repeats,
            seed=corpus_seed,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
            missing_rate=missing_rate,
            max_prompt_rows=max_prompt_rows,
            splits=cached_splits,
        )
    return combine_source_corpus_null_maxima(maxima)


@dataclass(frozen=True)
class SourceCorpusPowerReplicate:
    """Reusable result for one corpus, generator source eta, scenario, and seed."""

    corpus_name: str
    scenario: str
    generator_source_eta: float
    selected: tuple[Candidate, ...]
    selector_rejected: bool
    maximum_inner_gain: float
    endpoint_component_gains: np.ndarray
    prompt_block_ids: np.ndarray
    topology_component_advantage: np.ndarray | None
    call_loading: float
    persistent_loading: float
    request_loading: float


@dataclass(frozen=True)
class SourcePowerReplicate:
    scenario: str
    generator_source_eta_by_corpus: tuple[float, ...]
    corpus_names: tuple[str, ...]
    selected: tuple[tuple[Candidate, ...], ...]
    corpus_selector_rejected: tuple[bool, ...]
    familywise_rejected: bool
    maximum_inner_gain: float
    endpoint_component_gains: np.ndarray
    endpoint_mean_gains: np.ndarray
    inference_source_eta_grid: tuple[float, ...]
    endpoint_lower_bounds_by_source_eta: np.ndarray
    endpoint_worst_source_eta_lower_bounds: np.ndarray
    inference_identified: bool
    multiplier_critical_value: float
    multiplier_order_statistic_rank_one_based: int
    inference_prompt_blocks: tuple[int, ...]
    inference_source_regions: tuple[int, ...]
    topology_component_advantage: np.ndarray | None
    topology_truth_beats_derangement: bool | None
    promoted: bool
    call_loading: float
    persistent_loading: float
    request_loading: float


def run_source_corpus_power_replicate(
    design,
    scenario,
    source_eta,
    *,
    corpus_name,
    repeats,
    seed,
    null_threshold,
    gammas,
    rhos,
    mean_ridges,
    shrinkage,
    missing_rate,
    max_prompt_rows,
    splits=None,
):
    """Run the outcome/scoring work for one reusable corpus/source-eta world."""

    if corpus_name not in REQUIRED_SOURCE_CORPORA:
        raise ValueError("corpus_name must be exploratory or fresh")
    if not isinstance(design, SourceDesign):
        raise TypeError("design must be a SourceDesign")
    eta = _validate_source_eta(source_eta)
    geometry, splits = _source_replicate_context(
        design, seed, max_prompt_rows, splits=splits
    )
    field = draw_source_repeated_field(
        geometry,
        scenario,
        design,
        eta,
        repeats,
        derive_seed(seed, "field"),
        missing_rate=missing_rate,
    )
    states = draw_latent_states(design.component_count, derive_seed(seed, "states"))
    endpoint_gains = np.empty(
        (design.component_count, len(PRIMARY_ENDPOINTS)), dtype=float
    )
    topology_advantage = (
        np.empty(design.component_count, dtype=float)
        if scenario.truth_rho > 0.0 and not scenario.deranged_truth
        else None
    )
    selected_by_fold = []
    rejected_by_fold = []
    maxima = []
    call_loadings = []
    persistent_loadings = []
    request_loadings = []
    block_kernel = gamma_item_kernel(0.5, geometry.kernels)
    for fold in splits.outer:
        search = source_inner_candidate_search(
            field,
            geometry,
            fold,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        selected, rejected = select_strictly_calibrated(search, null_threshold)
        selected_by_fold.append(selected)
        rejected_by_fold.append(rejected)
        maxima.append(search.maximum_eligible_gain)
        nuisance = fit_repeat_nuisance(
            field,
            geometry,
            fold.train,
            fold.held,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
        )
        call_loadings.append(nuisance.call_loading)
        persistent_loadings.append(nuisance.persistent_loading)
        request_loadings.append(nuisance.request_loading)
        selected_kernel = (
            block_kernel
            if selected.is_block
            else gamma_item_kernel(selected.gamma, geometry.kernels)
        )
        residual = nuisance.evaluate_centered_means
        block_residual = component_marginal_quasi_nll(
            residual, nuisance, block_kernel, 0.0
        )
        selected_residual = component_marginal_quasi_nll(
            residual, nuisance, selected_kernel, selected.rho
        )
        held_states = states[fold.held]
        block_posterior = component_marginal_posterior_nll(
            residual, nuisance, block_kernel, 0.0, held_states
        )
        selected_posterior = component_marginal_posterior_nll(
            residual, nuisance, selected_kernel, selected.rho, held_states
        )
        endpoint_gains[fold.held, 0] = block_residual - selected_residual
        endpoint_gains[fold.held, 1] = block_posterior - selected_posterior
        if topology_advantage is not None:
            truth_kernel = scenario_item_kernel(geometry, scenario)
            topology_advantage[fold.held] = (
                component_marginal_quasi_nll(
                    residual,
                    nuisance,
                    deranged_item_kernel(truth_kernel),
                    scenario.truth_rho,
                )
                - component_marginal_quasi_nll(
                    residual,
                    nuisance,
                    truth_kernel,
                    scenario.truth_rho,
                )
            )
    prompt_block_ids = np.array(geometry.prompt_block_index, dtype=int, copy=True)
    endpoint_gains.setflags(write=False)
    prompt_block_ids.setflags(write=False)
    if topology_advantage is not None:
        topology_advantage.setflags(write=False)
    return SourceCorpusPowerReplicate(
        corpus_name=corpus_name,
        scenario=scenario.name,
        generator_source_eta=eta,
        selected=tuple(selected_by_fold),
        selector_rejected=bool(any(rejected_by_fold)),
        maximum_inner_gain=float(max(maxima)),
        endpoint_component_gains=endpoint_gains,
        prompt_block_ids=prompt_block_ids,
        topology_component_advantage=topology_advantage,
        call_loading=float(max(call_loadings)),
        persistent_loading=float(max(persistent_loadings)),
        request_loading=float(max(request_loadings)),
    )


def combine_source_power_corpus_replicates(
    records_by_corpus,
    designs_by_corpus,
    scenario,
    *,
    multiplier_seed,
    confidence=0.95,
    multiplier_draws=999,
    inference_source_eta_grid=SOURCE_ETA_GRID,
    prepared_multiplier_components_by_corpus=None,
):
    """Combine two cached corpus worlds into one joint primary event.

    The expensive outcome generation, nuisance fits, and endpoint scoring are
    corpus-local.  This combiner performs only the joint graph-aware max-t
    inference and conjunction gates, so the five worlds from each corpus can
    be reused across all 25 generator-source-eta pairs.
    """

    designs = _validated_design_mapping(designs_by_corpus)
    if not isinstance(records_by_corpus, Mapping) or set(records_by_corpus) != set(
        REQUIRED_SOURCE_CORPORA
    ):
        raise ValueError(
            "records_by_corpus must contain exactly exploratory and fresh"
        )
    corpus_records = tuple(
        records_by_corpus[corpus] for corpus in REQUIRED_SOURCE_CORPORA
    )
    for corpus, record, design in zip(
        REQUIRED_SOURCE_CORPORA, corpus_records, designs
    ):
        if not isinstance(record, SourceCorpusPowerReplicate):
            raise TypeError("every corpus record must be a SourceCorpusPowerReplicate")
        if record.corpus_name != corpus or record.scenario != scenario.name:
            raise ValueError("corpus power record identity mismatch")
        _validate_source_eta(record.generator_source_eta)
        endpoint_values = np.asarray(record.endpoint_component_gains, dtype=float)
        prompt_ids = np.asarray(record.prompt_block_ids)
        if (
            endpoint_values.shape
            != (design.component_count, len(PRIMARY_ENDPOINTS))
            or not np.isfinite(endpoint_values).all()
            or prompt_ids.shape != (design.component_count,)
            or len(record.selected) != OUTER_FOLDS
            or not math.isfinite(float(record.maximum_inner_gain))
            or not all(math.isfinite(float(value)) for value in (
                record.call_loading,
                record.persistent_loading,
                record.request_loading,
            ))
        ):
            raise ValueError("corpus power record does not align with its source design")
        if scenario.truth_rho > 0.0 and not scenario.deranged_truth:
            topology = np.asarray(record.topology_component_advantage, dtype=float)
            if topology.shape != (design.component_count,) or not np.isfinite(
                topology
            ).all():
                raise ValueError("corpus topology control is missing or malformed")
        elif record.topology_component_advantage is not None:
            raise ValueError("null/deranged corpus record cannot carry topology advantage")
    inference_source_eta_grid = _validate_source_eta_grid(
        inference_source_eta_grid
    )
    try:
        if prepared_multiplier_components_by_corpus is None:
            bounds = graph_aware_prompt_source_lower_bounds(
                tuple(
                    record.endpoint_component_gains
                    for record in corpus_records
                ),
                tuple(record.prompt_block_ids for record in corpus_records),
                designs,
                inference_source_eta_grid=inference_source_eta_grid,
                confidence=confidence,
                draws=multiplier_draws,
                seed=multiplier_seed,
            )
        else:
            if not isinstance(
                prepared_multiplier_components_by_corpus, Mapping
            ) or set(prepared_multiplier_components_by_corpus) != set(
                REQUIRED_SOURCE_CORPORA
            ):
                raise ValueError(
                    "prepared multiplier components must contain exactly "
                    "exploratory and fresh"
                )
            for corpus, record, design in zip(
                REQUIRED_SOURCE_CORPORA, corpus_records, designs
            ):
                prepared = prepared_multiplier_components_by_corpus[corpus]
                if not isinstance(
                    prepared, PreparedGraphAwareSourceCorpusMultiplier
                ):
                    raise TypeError(
                        "every prepared multiplier component must be a "
                        "PreparedGraphAwareSourceCorpusMultiplier"
                    )
                expected_input_identity = (
                    id(record.endpoint_component_gains),
                    id(record.prompt_block_ids),
                    id(design),
                )
                if (
                    prepared.corpus_name != corpus
                    or prepared.source_input_object_identity
                    != expected_input_identity
                    or prepared.multiplier_seed != int(multiplier_seed)
                    or prepared.draws != multiplier_draws
                    or prepared.inference_source_eta_grid
                    != inference_source_eta_grid
                ):
                    raise ValueError(
                        "prepared multiplier does not match its exact "
                        "record/design/inference identity"
                    )
            bounds = combine_prepared_graph_aware_source_corpus_multipliers(
                prepared_multiplier_components_by_corpus,
                confidence=confidence,
            )
        inference_identified = True
        lower_by_source_eta = bounds.lower_bounds_by_source_eta
        worst_lower = bounds.worst_source_eta_lower_bounds
        critical = bounds.critical_value
        order_rank = bounds.order_statistic_rank_one_based
        prompt_counts = bounds.prompt_block_counts
        source_counts = bounds.active_source_region_counts
    except MultiplierNotIdentifiedError:
        # An all-block selector can make the gain influence identically zero,
        # especially under null controls.  Preserve a finite, explicit
        # non-promotion record; unrelated input/numerical errors still escape.
        inference_identified = False
        lower_by_source_eta = np.zeros((
            len(REQUIRED_SOURCE_CORPORA),
            len(inference_source_eta_grid),
            len(PRIMARY_ENDPOINTS),
        ))
        worst_lower = np.zeros((
            len(REQUIRED_SOURCE_CORPORA), len(PRIMARY_ENDPOINTS)
        ))
        critical = 0.0
        order_rank = min(
            int(math.ceil(confidence * (multiplier_draws + 1))),
            multiplier_draws,
        )
        prompt_counts = tuple(
            len(np.unique(record.prompt_block_ids))
            for record in corpus_records
        )
        source_counts = tuple(
            int(np.count_nonzero(np.bincount(
                design.component_region_index,
                minlength=design.region_count,
            )))
            for design in designs
        )
    rejected = tuple(record.selector_rejected for record in corpus_records)
    familywise_rejected = bool(all(rejected))
    if scenario.truth_rho > 0.0 and not scenario.deranged_truth:
        topology_advantage = np.stack([
            record.topology_component_advantage for record in corpus_records
        ])
        topology_beats = bool(
            np.all(np.mean(topology_advantage, axis=1) > 0.0)
        )
    else:
        topology_advantage = None
        topology_beats = None
    promoted = bool(
        inference_identified
        and familywise_rejected
        and np.all(worst_lower > 0.0)
    )
    endpoint_gains = np.stack([
        record.endpoint_component_gains for record in corpus_records
    ])
    endpoint_means = endpoint_gains.mean(axis=1)
    for value in (
        endpoint_gains,
        endpoint_means,
        lower_by_source_eta,
        worst_lower,
        topology_advantage,
    ):
        if value is not None:
            value.setflags(write=False)
    return SourcePowerReplicate(
        scenario=scenario.name,
        generator_source_eta_by_corpus=tuple(
            record.generator_source_eta for record in corpus_records
        ),
        corpus_names=REQUIRED_SOURCE_CORPORA,
        selected=tuple(record.selected for record in corpus_records),
        corpus_selector_rejected=rejected,
        familywise_rejected=familywise_rejected,
        maximum_inner_gain=float(
            max(record.maximum_inner_gain for record in corpus_records)
        ),
        endpoint_component_gains=endpoint_gains,
        endpoint_mean_gains=endpoint_means,
        inference_source_eta_grid=inference_source_eta_grid,
        endpoint_lower_bounds_by_source_eta=lower_by_source_eta,
        endpoint_worst_source_eta_lower_bounds=worst_lower,
        inference_identified=inference_identified,
        multiplier_critical_value=critical,
        multiplier_order_statistic_rank_one_based=order_rank,
        inference_prompt_blocks=prompt_counts,
        inference_source_regions=source_counts,
        topology_component_advantage=topology_advantage,
        topology_truth_beats_derangement=topology_beats,
        promoted=promoted,
        call_loading=float(max(record.call_loading for record in corpus_records)),
        persistent_loading=float(
            max(record.persistent_loading for record in corpus_records)
        ),
        request_loading=float(
            max(record.request_loading for record in corpus_records)
        ),
    )


def run_source_power_replicate(
    designs_by_corpus,
    scenario,
    *,
    source_eta_by_corpus,
    repeats,
    seed,
    null_threshold,
    gammas=DEFAULT_GAMMAS,
    rhos=DEFAULT_RHOS,
    mean_ridges=DEFAULT_MEAN_RIDGES,
    shrinkage=0.05,
    confidence=0.95,
    multiplier_draws=999,
    inference_source_eta_grid=SOURCE_ETA_GRID,
    missing_rate=0.02,
    max_prompt_rows=MAX_PROMPT_ROWS,
    splits_by_corpus=None,
):
    """Run one joint world with corpus-specific source-dependence strengths."""

    designs = _validated_design_mapping(designs_by_corpus)
    etas = _validate_source_eta_by_corpus(source_eta_by_corpus)
    splits = _validated_optional_splits_mapping(splits_by_corpus)
    records = {
        corpus: run_source_corpus_power_replicate(
            design,
            scenario,
            eta,
            corpus_name=corpus,
            repeats=repeats,
            seed=derive_seed(seed, corpus),
            null_threshold=null_threshold,
            gammas=gammas,
            rhos=rhos,
            mean_ridges=mean_ridges,
            shrinkage=shrinkage,
            missing_rate=missing_rate,
            max_prompt_rows=max_prompt_rows,
            splits=cached_splits,
        )
        for corpus, design, eta, cached_splits in zip(
            REQUIRED_SOURCE_CORPORA, designs, etas, splits
        )
    }
    return combine_source_power_corpus_replicates(
        records,
        designs_by_corpus,
        scenario,
        multiplier_seed=derive_seed(seed, "joint-prompt-source-multiplier"),
        confidence=confidence,
        multiplier_draws=multiplier_draws,
        inference_source_eta_grid=inference_source_eta_grid,
    )


def aggregate_source_power_records(records: Sequence[SourcePowerReplicate]):
    """Aggregate compact Stage-A records without retaining component arrays."""

    records = tuple(records)
    if not records:
        raise ValueError("at least one source power record is required")
    if len({
        (row.scenario, tuple(row.generator_source_eta_by_corpus))
        for row in records
    }) != 1:
        raise ValueError(
            "source power records must share one scenario and generator source-eta pair"
        )
    if any(row.corpus_names != REQUIRED_SOURCE_CORPORA for row in records):
        raise ValueError("source power records have incompatible corpus order")
    if any(
        len(row.generator_source_eta_by_corpus) != len(REQUIRED_SOURCE_CORPORA)
        or any(
            eta not in SOURCE_ETA_GRID
            for eta in row.generator_source_eta_by_corpus
        )
        for row in records
    ):
        raise ValueError(
            "source power records have an invalid generator source-eta pair"
        )
    if len({row.inference_source_eta_grid for row in records}) != 1:
        raise ValueError("source power records have incompatible inference eta grids")
    selected = [
        candidate
        for row in records
        for corpus_selected in row.selected
        for candidate in corpus_selected
    ]
    topology = [
        row.topology_truth_beats_derangement
        for row in records
        if row.topology_truth_beats_derangement is not None
    ]
    return {
        "scenario": records[0].scenario,
        "generator_source_eta_by_corpus": {
            corpus: records[0].generator_source_eta_by_corpus[corpus_index]
            for corpus_index, corpus in enumerate(REQUIRED_SOURCE_CORPORA)
        },
        "replicates": len(records),
        "both_corpus_selector_rejections": int(sum(
            row.familywise_rejected for row in records
        )),
        "both_corpus_selector_rejection_rate": float(np.mean([
            row.familywise_rejected for row in records
        ])),
        "joint_source_primary_events": int(sum(row.promoted for row in records)),
        "joint_source_primary_event_rate": float(np.mean([
            row.promoted for row in records
        ])),
        "inference_nonidentified_replicates": int(sum(
            not row.inference_identified for row in records
        )),
        "all_replicates_inference_identified": bool(all(
            row.inference_identified for row in records
        )),
        "endpoint_mean_gain_per_scalar": {
            corpus: {
                endpoint: summary(
                    row.endpoint_mean_gains[corpus_index, endpoint_index]
                    for row in records
                )
                for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS)
            }
            for corpus_index, corpus in enumerate(REQUIRED_SOURCE_CORPORA)
        },
        "worst_source_eta_simultaneous_lower_bound": {
            corpus: {
                endpoint: summary(
                    row.endpoint_worst_source_eta_lower_bounds[
                        corpus_index, endpoint_index
                    ]
                    for row in records
                )
                for endpoint_index, endpoint in enumerate(PRIMARY_ENDPOINTS)
            }
            for corpus_index, corpus in enumerate(REQUIRED_SOURCE_CORPORA)
        },
        "inference_source_eta_grid": list(
            records[0].inference_source_eta_grid
        ),
        "multiplier_max_t_critical_value": summary(
            row.multiplier_critical_value for row in records
        ),
        "topology_truth_beats_derangement_rate": (
            float(np.mean(topology)) if topology else None
        ),
        "selected_gamma_counts_across_outer_folds": {
            str(value): int(sum(candidate.gamma == value for candidate in selected))
            for value in sorted({candidate.gamma for candidate in selected})
        },
        "selected_rho_item_counts_across_outer_folds": {
            str(value): int(sum(candidate.rho == value for candidate in selected))
            for value in sorted({candidate.rho for candidate in selected})
        },
        "maximum_call_loading": float(max(row.call_loading for row in records)),
        "maximum_persistent_loading": float(
            max(row.persistent_loading for row in records)
        ),
        "maximum_request_loading": float(
            max(row.request_loading for row in records)
        ),
    }

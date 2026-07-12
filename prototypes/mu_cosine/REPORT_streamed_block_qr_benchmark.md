# Batched measurement blocks with the joint square-root / QR conditioner

Run 2026-07-11. This report benchmarks the sequential regime left open by
`REPORT_joint_square_root_qr_benchmark.md`.

## Result in one paragraph

Batching several available measurement blocks into **one stacked Householder QR** is substantially faster than
launching one QR per block. CUDA becomes worthwhile for QR when a fixed design has thousands of independent
right-hand sides: at `n=128`, 32 measurements per block, 16 blocks, and `B=4096`, full stacked QR fell from
`10.392 ± 0.197 ms` on CPU to `3.939 ± 0.034 ms` on CUDA. It did **not** cross over at `B=128`, and batched QR
of 128 distinct designs was slower on the local GTX 1660 SUPER than CPU QR. Dense normal-equation updates were
usually faster, but in the conditioning stress test their float32 Cholesky failed in one regime and incurred
`1.50e-4` relative solution error in another, while Householder QR stayed near `1e-6`. The resulting policy is:
stack blocks that are already available, stream only genuinely online/adaptive blocks, retain QR for its
stability, and use CUDA after measuring the actual shared-versus-distinct design shape.

This square-root conditioner is not `JointPosterior`. `JointPosterior` is a learned nonlinear decision model;
the conditioner is a numerical backend for a correlated linear-Gaussian posterior. They can be used together.

## Four dimensions, not two

The benchmark uses separate symbols for quantities that are easy to conflate:

| symbol | meaning | values exercised |
|---|---|---|
| `n` | latent-state dimension | `2, 32, 128` |
| `m_block` | measurements/judges in one block | `32` |
| `T` | blocks assimilated into one posterior | selected `1,4,16`; CLI also exposes `2,8` |
| `B` | independent posterior systems processed together | `128`, plus the `4096` crossover cell |
| `M_total` | total likelihood rows, `T * m_block` | `32,128,512` |

Thus the proposed `n=128,m=32` case becomes a `(n + Tm_block) × n` information pre-array when all blocks are
stacked. `B` is a separate CUDA batch/RHS axis.

Two design regimes matter:

- **shared:** all `B` systems use the same block coefficients and differ only in their right-hand sides. One
  factorization can serve all systems.
- **distinct:** every system has its own block coefficients, as can happen when covariance or observation
  geometry depends on the item. CUDA must perform `B` distinct factorizations.

## Statistical construction and algorithms

Each synthetic block starts with a dense within-block conditional covariance `Rc_t`. It is Cholesky-whitened to

```text
A_t = chol(Rc_t)^-1 J_t
b_t = chol(Rc_t)^-1 r_t
```

and conditional cross-block covariance is exactly zero. Whitening and covariance estimation are outside the
timed region. That isolates the conditioner; it also means this benchmark does **not** yet test semantic- or
graph-distance cross-block correlation.

All algorithms start with `P^-1 = U.T @ U` and zero information RHS:

1. **`full_stacked_qr`:** one Householder QR of `stack([U,A_1,...,A_T])`.
2. **`streamed_qr`:** `T` Householder QR calls, carrying `(U_t,z_t)` into the next block.
3. **`dense_recompute`:** accumulate `Lambda += A_t.T @ A_t` and `eta += A_t.T @ b_t`, Cholesky-refactoring
   the normal equations after every block. This is a speed baseline, not a square-root-stability method.
4. **`cached_dense_full`:** for a shared, completely known design only, factor the final normal equations once
   and apply new RHS batches. It does not produce the intermediate online roots.

The stacked and streamed QR paths produce the same posterior in exact arithmetic. Stacking wins when all blocks
are already present because it replaces `T` factorizations and launches with one larger operation. Streaming is
still necessary when a block arrives later, depends on an earlier posterior, or an intermediate root is itself
the required output.

## Hardware and timing protocol

- CPU: Intel Core i7-10700KF, eight PyTorch threads.
- GPU: NVIDIA GeForce GTX 1660 SUPER, 6 GiB.
- PyTorch 2.4.1+cu121; CUDA runtime 12.1; WSL2.
- Nominal workload: float32, target prior/within-block covariance condition number 10, conditional-noise scale
  1, seed `20260711`.
- CUDA timing synchronizes around every trial. Tables report median ± median absolute deviation (MAD).
- Input H2D and result D2H copies are timed separately and excluded from trajectory latency. The current harness
  pre-casts on the host so H2D measures transfer rather than transfer plus a CUDA-only dtype conversion.
- Peak CUDA memory is incremental temporary allocation, not the allocator's reserved pool; it excludes any
  compiled object already resident at the start of a conditioning trial.
- The final `n=128,B=128` CUDA cells use 11 trials and CPU cells use 7; the smaller-state table uses 11. The
  `B=4096` crossover uses 11 trials. All have warmups. Exact commands appear below.

### Authoritative-run policy

The compute tables use the **final recorded runs**: compute receives the configured warmups and CUDA's caching
allocator is not emptied between timed trials. Peak allocation is measured with `reset_peak_memory_stats`
without changing that steady-state timing protocol.

These runs supersede exploratory figures produced while the harness still called `empty_cache()` before each
trial and before transfer timing had a warmup. In particular, do not mix the superseded
`2.934 / 196.941 ms` `B=128` CUDA values or the `10.666 / 3.967 ms`, `H2D=4.375 ms` `B=4096` values with the
tables below. The authoritative corresponding compute values are `3.424 / 193.247 ms` and
`10.392 / 3.939 ms`.

The recorded `H2D=4.855 ms` value also included float64-to-float32 conversion. A post-review harness correction
now pre-casts identically for CPU and CUDA, so that value is retained below only as a conservative historical
upper bound; the current command must be rerun on a CUDA host for a transfer-only number. Compute timings are
unaffected by this correction.

## `n=128`, `m_block=32`, `B=128`

### Shared design

Trajectory median ± MAD, milliseconds. Cached dense is shown as `setup + condition`.

| `T` | CPU stacked QR | CUDA stacked QR | CPU streamed QR | CUDA streamed QR | CPU dense recompute | CUDA dense recompute | CPU cached dense | CUDA cached dense |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.719 ± .005 | 3.147 ± .088 | 0.749 ± .017 | 2.980 ± .090 | 0.409 ± .010 | 0.767 ± .007 | .136 + .165 | .415 + .311 |
| 4 | 0.873 ± .006 | 2.743 ± .179 | 3.075 ± .110 | 10.474 ± .342 | 1.484 ± .036 | 2.382 ± .087 | .173 + .205 | .410 + .329 |
| 16 | **1.273 ± .024** | 3.424 ± .095 | 11.033 ± .495 | 39.879 ± 1.281 | 5.731 ± .244 | 9.574 ± .637 | .177 + .218 | .406 + .335 |

At this batch size the CPU wins every shared-design QR cell. At `T=16`, stacking is 8.7× faster than streaming
on CPU and 11.6× on CUDA. The cached-dense row is a static lower bound, not a matched cache comparison: this
sequential harness does not yet time cached full/streamed QR chains. Compare `setup + condition` for a one-shot
call; use `REPORT_joint_square_root_qr_benchmark.md` for the matched repeated fixed-design comparison.

### Distinct design per system

| `T` | CPU stacked QR | CUDA stacked QR | CPU streamed QR | CUDA streamed QR | CPU dense recompute | CUDA dense recompute |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.356 ± 1.412 | 113.350 ± .362 | 24.472 ± .513 | 113.111 ± .261 | 22.207 ± 4.185 | **1.625 ± .038** |
| 4 | 34.799 ± .750 | 119.602 ± .466 | 109.855 ± 11.635 | 454.403 ± 2.043 | 45.038 ± 1.640 | **6.152 ± .103** |
| 16 | **66.145 ± .878** | 193.247 ± .788 | 422.071 ± 11.815 | 1804.537 ± 4.470 | 177.832 ± 4.051 | **23.828 ± .187** |

The local GPU's batched `geqrf` path is a poor fit for 128 distinct tall-skinny factorizations: CPU
Householder wins decisively. CUDA does accelerate the dense batched Cholesky baseline. This is a backend and
hardware result, not evidence against QR's numerical role.

For the distinct `T=16` CUDA cell, the historical cast-plus-H2D upper bound is `15.097 ± .239 ms`; D2H is about
`1.6 ms`. Peak incremental
allocation is 160.94 MiB for stacked QR, 46.61 MiB for streamed QR, and 32.19 MiB for dense recomputation.
Streaming trades substantially more time for bounded memory as `T` grows.

## The shared-design CUDA crossover: `B=4096`

At `n=128,m_block=32,T=16`, increasing the shared RHS batch from 128 to 4096 produces the expected GPU
crossover:

| algorithm | CPU trajectory ms | CUDA trajectory ms | CUDA peak MiB |
|---|---:|---:|---:|
| full stacked QR | 10.392 ± .197 | **3.939 ± .034** | 34.95 |
| streamed QR | 54.951 ± 2.336 | **39.146 ± .407** | 17.31 |
| dense recompute | 54.164 ± 3.870 | **20.464 ± .066** | 10.19 |
| cached dense full | .183 setup + 2.812 condition | **.492 setup + 1.220 condition** | 6.25 |

The historical cast-plus-input-transfer upper bound is `4.855 ± .171 ms`; output transfer is `.74–.93 ms`.
Even with that conservative surcharge, stacked QR barely wins end-to-end if every batch begins and ends on the
host; device residency materially strengthens the CUDA case. The current pre-cast harness needs a CUDA rerun
before quoting a transfer-only crossover. Cached dense loses its end-to-end CUDA advantage under the historical
cast-plus-transfer protocol.

## Smaller states

The selected distinct-design grid shows the same qualitative result at `B=128,T=16`:

| `n` | CPU stacked QR ms | CUDA stacked QR ms | CPU streamed QR ms | CUDA streamed QR ms | CPU / CUDA dense recompute ms |
|---:|---:|---:|---:|---:|---:|
| 2 | **.808** | 5.958 | **10.387** | 54.136 | **2.388** / 7.739 |
| 32 | **14.749** | 46.922 | **64.750** | 420.016 | **9.604** / 10.287 |
| 128 | **66.145** | 193.247 | **422.071** | 1804.537 | 177.832 / **23.828** |

There is no general rule that a larger pre-array automatically makes CUDA QR faster. Shared reflector reuse,
the number of independent RHSs, distinct versus common designs, GPU generation, and transfer all matter.

## Numerical behavior

Every output is compared with a float64 dense full-trajectory reference. The CSV reports:

- relative final-solution error;
- relative root error against the positive-diagonal float64 Cholesky root;
- relative Gram error `||U.T U - Lambda_ref|| / ||Lambda_ref||`; and
- solution and Gram changes after reversing block order.

In the nominal `n=128,B=128,T=16` float32 runs, QR solution errors were below `7.3e-7`, Gram errors below
`3.0e-7`, and reverse-order solution changes below `1.1e-6`. At `B=4096`, streamed QR's solution error was
`1.06e-6` and its reverse-order change was `1.50e-6`.

A focused `condition=1e6` CUDA stress run demonstrates why the root method remains useful:

| noise scale | dtype | stacked QR solution / Gram error | streamed QR solution / Gram error | dense solution / Gram error |
|---:|---|---:|---:|---:|
| .001 | float32 | `5.73e-7 / 1.41e-7` | `1.44e-6 / 4.11e-7` | **Cholesky failed** |
| 1000 | float32 | `8.18e-7 / 1.96e-7` | `2.07e-6 / 8.13e-7` | `1.50e-4 / 1.83e-7` |
| .001 | float64 | `1.20e-14 / 3.90e-16` | `1.23e-14 / 8.07e-16` | `1.47e-14 / 3.15e-16` |
| 1000 | float64 | `2.35e-13 / 5.23e-16` | `2.35e-13 / 1.44e-15` | `2.88e-13 / 2.79e-16` |

The dense baseline forms normal equations. Its tiny Gram error does not guarantee an accurate solution when
conditioning is poor, and its mathematically positive precision can become non-positive in float32. The
benchmark records per-algorithm execution failures as CSV rows rather than aborting that cell.

## What “blocks on the diagonal” permits

Exact independent streaming requires block diagonality of the **conditional** covariance

```text
Rc = R - C.T @ P^-1 @ C,
```

not merely raw `R`. Dense interactions among the 32 judges inside one block are already permitted and are
absorbed by that block's whitening factor. Off-block semantic or graph-distance correlation makes `Rc` dense
and cannot be ignored without changing the statistical model.

A good next statistical extension is a positive-semidefinite structured residual model, fit on held-out
calibration residuals, for example

```text
Rc = D + K_item(graph_distance, semantic_distance) ⊗ Sigma_judge.
```

Use a PSD graph diffusion kernel or embedding RBF rather than an arbitrary distance-to-correlation rule. A
low-rank shared judge/corpus-drift term is especially attractive: augmenting the latent state with those
factors can restore conditional block independence and preserve streamed QR. Compare full dense whitening,
the structured model, and a block-diagonal approximation on posterior KL/calibration—not just runtime.

This covariance experiment is distinct from measuring whether graph-judge supervision helps a model learn a
dataset's entities and topology. That requires a matched training/adaptation ablation.

## Reproduction

Lightweight default (`n=128`, `T=1,4`, `B=1,128`, distinct float32 design; CPU and CUDA when available):

```bash
python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py
```

The full shape grid is exposed but intentionally not the default:

```bash
python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --full-grid --measurements-per-block 32 --devices cpu,cuda \
  --warmups 5 --trials 21 --transfer-trials 11 --cpu-threads 8
```

The `n=128,B=128` reported cells were produced by the following CPU and CUDA runs:

```bash
python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 128 --measurements-per-block 32 --block-counts 1,4,16 \
  --batch-sizes 128 --design-modes shared,distinct --dtypes float32 \
  --condition-numbers 10 --noise-scales 1 \
  --algorithms full_stacked_qr,streamed_qr,dense_recompute,cached_dense_full \
  --devices cpu --warmups 2 --trials 7 --transfer-trials 1 \
  --inner-repeats 1 --cpu-threads 8

python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 128 --measurements-per-block 32 --block-counts 1,4,16 \
  --batch-sizes 128 --design-modes shared,distinct --dtypes float32 \
  --condition-numbers 10 --noise-scales 1 \
  --algorithms full_stacked_qr,streamed_qr,dense_recompute,cached_dense_full \
  --devices cuda --warmups 5 --trials 11 --transfer-trials 7 \
  --inner-repeats 1 --cpu-threads 8

python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 2,32 --measurements-per-block 32 --block-counts 16 \
  --batch-sizes 128 --design-modes distinct --dtypes float32 \
  --condition-numbers 10 --noise-scales 1 \
  --algorithms full_stacked_qr,streamed_qr,dense_recompute \
  --devices cpu,cuda --warmups 5 --trials 11 --transfer-trials 7 \
  --inner-repeats 1 --cpu-threads 8
```

The `B=4096` crossover and numerical stress commands were:

```bash
python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 128 --measurements-per-block 32 --block-counts 16 \
  --batch-sizes 4096 --design-modes shared --dtypes float32 \
  --condition-numbers 10 --noise-scales 1 \
  --algorithms full_stacked_qr,streamed_qr,dense_recompute,cached_dense_full \
  --devices cpu,cuda --warmups 5 --trials 11 --transfer-trials 7 \
  --inner-repeats 1 --cpu-threads 8

python3 -u prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 128 --measurements-per-block 32 --block-counts 16 \
  --batch-sizes 128 --design-modes shared --dtypes float32,float64 \
  --condition-numbers 1000000 --noise-scales 0.001,1000 \
  --algorithms full_stacked_qr,streamed_qr,dense_recompute \
  --devices cuda --warmups 1 --trials 2 --transfer-trials 2 \
  --inner-repeats 1 --cpu-threads 8
```

Validation performed:

```bash
python3 -m py_compile prototypes/mu_cosine/benchmark_streamed_block_qr.py

python3 prototypes/mu_cosine/benchmark_streamed_block_qr.py \
  --state-dims 3 --measurements-per-block 4 --block-counts 2 \
  --batch-sizes 2 --design-modes shared,distinct --dtypes float32 \
  --condition-numbers 10 --noise-scales 1 --devices cpu \
  --warmups 1 --trials 1 --transfer-trials 1 --inner-repeats 1 \
  --cpu-threads 1
```

All smoke cells completed with `status=ok`; the expected dense float32 algorithm failure was captured as
`failed_LinAlgError`. Per-algorithm execution failures are emitted as CSV status rows after a problem and its
reference have been constructed. Problem generation, reference construction, or device-transfer/OOM failures
can still abort the sweep and should be handled by the outer job runner. The complete condition/scale grid is
deliberately opt-in because it multiplies an already large shape grid.

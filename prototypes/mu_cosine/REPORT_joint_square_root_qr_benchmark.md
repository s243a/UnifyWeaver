# Fixed-design Gaussian conditioning: dense-gain versus square-root/QR

Run 2026-07-10. This is a numerical throughput benchmark, not a statistical comparison: the PyTorch backend
is tested against the NumPy Householder implementation and the dense correlated Gaussian conditioner before
timing.

## Implementation under test

For one fixed `(U,H,R,C)` design, `CompiledCorrelatedConditionerTorch`:

1. decorrelates the measurement block through `Rc = R - C^T P^-1 C` and `J = H + C^T P^-1`;
2. Cholesky-whitens `(J,Rc)`;
3. calls `torch.geqrf` once on `stack([U, L^-1 J])`, retaining compact Householder reflectors;
4. treats a batch of innovations as RHS columns, applies `Q^T` with `torch.ormqr`, and performs one triangular
   solve.

`Q` is never formed. A generic distinct-design batch and a sequential state that threads `(U_post,z_post)`
are also implemented, but only the fixed-design path can share the factorisation across rows.

For the matched static baseline, `CompiledDenseGainConditionerTorch` derives the same conditional `(J,Rc)`
model, computes the dense correlated gain and posterior covariance once, and batches only the per-row affine
mean update. Here “compiled” means **design-cached**, not `torch.compile` or kernel fusion. It cannot reuse that
gain after a sequential block changes the posterior.

## Hardware and protocol

- CPU: Intel Core i7-10700KF, 8 WSL-visible logical CPUs; `torch.set_num_threads(8)`.
- GPU: NVIDIA GeForce GTX 1660 SUPER, 6 GiB, driver 591.74.
- PyTorch 2.4.1+cu121; CUDA runtime 12.1; float32.
- Selected cells were run in isolation with 20 warmups, 500 conditioning repeats, and 50 factorisation
  repeats; explicit CUDA synchronisation around timings.
- Timings are on-device compute point measurements from one process run. They exclude host/device transfer and
  report a mean over repeats, not median/MAD across independent trials.
- `compile_ms` and `condition_ms` are deliberately separate. Rows/s measures only repeated conditioning after
  compilation. At the time of this run, dense compilation checked/regularised a subtraction-form posterior
  covariance. The implementation now uses the algebraically equivalent Joseph form to avoid that catastrophic
  subtraction; the conditioning timings are unaffected, but the historical setup timings were not rerun.

The matched baseline, `CompiledDenseGainConditionerTorch`, validates the same Schur-complement model and
precomputes the correlated gain `K` and posterior covariance. Each row then needs only the affine innovation
update. This is the appropriate static-design comparator; QR's distinct purpose is carrying an updated
precision root through changing or sequential blocks.

Selected synchronized results:

| n | m | rows | CPU QR compile / condition ms | CPU dense compile / condition ms | CUDA QR compile / condition ms | CUDA dense compile / condition ms |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 1 | 0.506 / 0.220 | **0.459 / 0.058** | 1.939 / 0.453 | 1.818 / 0.116 |
| 2 | 4 | 4096 | 0.434 / 0.596 | **0.422 / 0.113** | 1.523 / 0.520 | 1.945 / 0.132 |
| 2 | 32 | 4096 | 0.474 / 0.992 | 0.465 / 0.256 | 2.722 / 0.534 | **2.679 / 0.129** |
| 32 | 32 | 4096 | 0.724 / 2.106 | 1.037 / 0.363 | 2.386 / 0.543 | **3.667 / 0.129** |
| 128 | 32 | 4096 | 0.800 / 5.694 | 1.532 / 0.734 | 3.273 / 1.121 | **5.402 / 0.131** |

## Interpretation

- The dense-gain baseline is faster than QR on both devices in every tested fixed-design cell. For the current
  `n=2,m=4` problem it is 3.8x faster than QR on CPU at batch one and 5.3x faster at batch 4096. CPU dense is
  also faster than CUDA dense there, so the current conditioner has no GPU throughput case.
- Large same-design batches do benefit from GPU parallelism, but it is not a QR-specific win. At
  `n=32,m=32,batch=4096`, CUDA dense takes 0.129 ms versus CPU dense 0.363 ms; at `n=128,m=32`, CUDA dense takes
  0.131 ms versus CPU dense 0.734 ms.
- QR remains useful when the deliverable is the updated precision root or later blocks change the posterior.
  In that sequential regime `(U_post,z_post)` must feed the next factorization; a static cached gain cannot be
  reused. All RHS blocks must stay in one coordinate system; recentering requires resetting `z` and shifting
  the later likelihood RHS. That regime needs its own matched end-to-end benchmark.
- Constant `H,P,C,R` is a performance opportunity, not a correctness shortcut: compile the dense gain for
  static rows. Prefer root-threading QR when the design/posterior changes and its numerical behavior earns the
  extra work.
- These crossover points are hardware/runtime-specific. Rebenchmark on deployment hardware and include data
  transfer if inputs do not already reside on the device.
- The table reports one timed run per cell after warmup, not multi-trial dispersion. Treat the exact speedups
  as local point measurements; the qualitative small-state/large-batch crossover reproduced, but CPU
  contention can move the ratios materially.

Reproduction:

```bash
python3 -m pytest -q -s \
  prototypes/mu_cosine/test_joint_square_root_conditioner.py \
  prototypes/mu_cosine/test_joint_square_root_conditioner_torch.py

python3 -u prototypes/mu_cosine/benchmark_joint_square_root_qr.py \
  --state-dims 2,32,128 --measurement-dims 4,32 --batch-sizes 1,4096 \
  --devices cpu,cuda --dtype float32 --warmups 20 --repeats 500 \
  --compile-repeats 50 --cpu-threads 8
```

The combined NumPy/PyTorch suite passes 29 tests, including dense-gain/QR float32/float64 parity, CUDA parity,
CPU/CUDA distinct-design batches, nonzero-`C` block-diagonal-`Rc` streaming, and CPU/CUDA fixed-coordinate root
threading plus recentered sequential blocks. Compiled conditioners also snapshot their fixed design so later
caller-side mutation cannot silently mix stale factors with changed `H`, `R`, or `C` inputs.

Next benchmark: compare end-to-end latency, memory, and numerical drift under genuinely sequential block
workloads where each update changes the carried posterior root. Add multi-trial median/MAD timings, host/device
transfer, peak-memory accounting, and a log-spaced scale/condition-number sweep. The static-design decision is
already clear: use the compiled dense gain.

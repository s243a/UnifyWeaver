# Semantic Search Implementation Plan

## Phase 1: Core Framework (Completed)
- [x] Create `semantic_compiler.pl` with `declare_semantic_provider/2`, `compile_semantic_call/4`.
- [x] Implement `semantic_dispatch/5` multifile hooks for Go, Python, Rust, and C#.
- [x] Integrate with `go_target.pl` (`compile_semantic_rule_go/4`).
- [x] Add multi-target verification tests.

## Phase 2: Hardware-Aware Code Generation (Completed)
- [x] Implement `device(Device)` option parsing in all targets.
- [x] GPU-specific initialization: Python (CUDA), Go (hugot WithGPU), C# (DirectML), Rust (candle CUDA / ONNX CUDA).
- [x] CPU fallback in generated code: Go (err retry), Python (torch check), C# (try/catch), Rust (unwrap_or_else).
- [x] Tests for GPU/CPU code generation paths.

## Phase 2b: Fuzzy Logic Integration (Completed)
- [x] Add `fuzzy_dispatch/3` multifile hook to `semantic_compiler.pl`.
- [x] Add `is_fuzzy_predicate/1` and `compile_fuzzy_call/3` API.
- [x] Wire Python fuzzy target (`python_fuzzy_target.pl`) through generic dispatch.
- [x] Implement Go fuzzy dispatch: f_and, f_or, f_dist_or, f_union, f_not, blend_scores, top_k.
- [x] Tests for fuzzy recognition, Go codegen, Python dispatch.

## Phase 3: Runtime Integration & Data Sources
- [ ] Extend `input_source.pl` to handle vector index files for different targets.
- [ ] Add support for `semantic_search/4` (options including threshold, custom model).
- [ ] Integrate with `cross_runtime_pipeline.pl` to allow multi-language semantic stages.
- [ ] Implement Rust and C# fuzzy dispatch (core operations).
- [ ] Go fuzzy batch operations (slice-based vectorization).

## Phase 4: Extended Target Coverage & Polish
- [ ] Implement `semantic_dispatch` for Elixir (via Bumblebee/NX).
- [ ] Add Python MPS (Apple Silicon) device support.
- [ ] Create documentation examples for GPU deployment.
- [ ] Final performance benchmarking across providers and devices.

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

## Phase 2c: Full Fuzzy Target Coverage (Completed)
- [x] Implement Rust fuzzy dispatch: f_and, f_or, f_dist_or, f_union, f_not, blend_scores, top_k.
- [x] Implement C# fuzzy dispatch: all core ops + LINQ-based blend_scores and top_k.
- [x] Go batch fuzzy operations: f_and_batch, f_or_batch, f_dist_or_batch, f_union_batch.
- [x] Tests for Rust (AND/OR/NOT), C# (AND/OR/NOT), Go batch (AND/OR).

## Phase 3: Runtime Integration & Data Sources (Completed)
- [x] Extend `input_source.pl` with `vector_db(Path)` / `vector_db(Path, Format)` modes.
- [x] Add `resolve_vector_source/2` and `vector_db_init_code/3` for all 4 targets.
- [x] Support `semantic_search/4` via `extract_search_options/2` and `merge_provider_options/3`.
- [x] Inline options (threshold, model, index) override provider config at compile time.
- [x] Integrate `cross_runtime_pipeline.pl` with semantic/fuzzy dispatch for Rust/C# stages.
- [x] Rust batch fuzzy operations (f_and_batch, f_or_batch, f_dist_or_batch, f_union_batch) via Vec<f64> iterators.
- [x] C# batch fuzzy operations via LINQ Enumerable/Zip patterns.

## Phase 4: Extended Target Coverage & Polish
- [ ] Implement `semantic_dispatch` for Elixir (via Bumblebee/NX).
- [ ] Add Python MPS (Apple Silicon) device support.
- [ ] Create documentation examples for GPU deployment.
- [ ] Final performance benchmarking across providers and devices.

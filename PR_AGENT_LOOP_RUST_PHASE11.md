# PR: Rust Phase 11 — Proot sandbox + integration tests

**Branch:** `feat/agent-loop-rust-phase11`
**Base:** `main`

## Summary

Implements proot filesystem isolation for the Rust target (matching Python's Layer 4 security) and adds 38 Rust-native integration tests that compile and run with `cargo test`.

### 1. Proot sandbox (proot_sandbox.rs)

- `ProotConfig` struct: allowed_dirs, readonly_binds, kill_on_exit, termux_prefix, extra_flags, redirect_home, dry_run
- `ProotSandbox` struct: working_dir, proot_path discovery, availability caching
- `is_available()` — searches PATH for proot binary
- `wrap_command()` — wraps commands with proot flags, bind mounts (Termux prefix, /proc, /dev, /system, working dir, allowed dirs, home redirect)
- `build_env_overrides()` — PROOT_NO_SECCOMP=1, LD_PRELOAD=""
- `status()` — diagnostic info via ProotStatus struct
- Shell quoting via `quote_parts()` and `shell_escape_sq()` helpers
- `shellexpand_home()` — expands `~/` to `$HOME`

### 2. Security profile extension

- `SecurityProfileSpec` gains `proot_isolation: bool` field
- `compile_component` for Rust security profiles now emits `proot_isolation` field
- All 4 profiles (open, cautious, guarded, paranoid) default to `proot_isolation: false`

### 3. ToolHandler proot wiring

- `ToolHandler.proot: Option<ProotSandbox>` field
- `enable_proot()` method creates ProotSandbox with allowed dirs
- `handle_bash()` wraps commands in proot when sandbox is enabled
- Proot env overrides applied to subprocess execution
- CLI `--proot` flag activates sandbox in main loop

### 4. Integration tests (38 tests)

Generated to `tests/integration_tests.rs`, compiled and run with `cargo test`:

| Category | Tests | Scope |
|----------|-------|-------|
| Context manager | 8 | add, clear, edit, undo, delete, range, tokens, history, tool_result |
| Cost tracker | 2 | initial state, record usage |
| Security profiles | 4 | existence, open/paranoid validation, blocked paths |
| Tool handler | 7 | path blocking, command blocking, approvals, execute bash/read/unknown, blocked rejection |
| Proot sandbox | 5 | config default, creation, env overrides, custom config, handler field |
| Config data | 2 | CLI args, AgentConfig defaults |
| Types | 3 | Message creation, ToolCall serialization roundtrip, ToolResult fields |
| E2E flows | 4 | bash echo flow, write+read flow, security blocks, multi-round context |

## Test plan

- [x] 740 Prolog tests pass (17 new Phase 11 assertions), 1 pre-existing env-specific failure
- [x] 38 `cargo test` integration tests pass
- [x] `cargo check` — 0 errors, 0 warnings
- [x] Only expected files changed
- [x] Rust fragment count updated (28 → 30)

## Metrics

| Metric | Phase 10 | Phase 11 |
|--------|----------|----------|
| Prolog tests | 723 | 740 (+17) |
| Cargo tests | 0 | 38 (+38) |
| rust_fragment/2 facts | 28 | 30 (+2: proot_sandbox, integration_tests) |
| Generated files | 14 | 15 (+proot_sandbox.rs) + tests/ |
| Proot sandbox | Python-only | Python + Rust |
| SecurityProfileSpec fields | 2 | 3 (+proot_isolation) |

## Files changed

| File | Changes |
|------|---------|
| `agent_loop_module.pl` | New proot_sandbox + integration_tests fragments; extended tool_handler_struct (proot field, enable_proot); updated handle_bash for proot wrapping; proot CLI wiring in main_loop; generate_rust_proot_sandbox + generate_rust_integration_tests generators |
| `agent_loop_components.pl` | Updated SecurityProfileSpec compile_component to emit proot_isolation field |
| `test_agent_loop.pl` | 3 new test predicates (17 assertions); updated fragment count (30); fixed security profile substring tests |
| `README.md` | Updated test count, fragment count (30), Rust feature list, parity table (proot complete, integration tests added) |
| `generated/rust/src/proot_sandbox.rs` | NEW — ProotConfig, ProotSandbox, wrap_command, env overrides |
| `generated/rust/src/tool_handler.rs` | proot field, enable_proot, bash command wrapping |
| `generated/rust/src/security.rs` | proot_isolation field in SecurityProfileSpec and profiles |
| `generated/rust/src/lib.rs` | pub mod proot_sandbox |
| `generated/rust/src/main.rs` | --proot CLI flag handling |
| `generated/rust/tests/integration_tests.rs` | NEW — 38 integration tests |

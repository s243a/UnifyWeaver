# PR: Rust Phase 10 — Full templates, persistence, retry wiring, spinner, rich display

**Branch:** `feat/agent-loop-rust-phase10`
**Base:** `main`

## Summary

Polishes the Rust target to full template parity (16 built-ins), adds template persistence to `~/.agent-loop/templates.json`, wires `retry_with_backoff` into actual API calls, and adds terminal spinner + enhanced token display during API responses.

### 1. Full built-in templates (6 → 16)

- Added 10 new templates: convert, translate, simplify, debug, optimize, regex, sql, bash, git, doc
- Matches Python target's full template set
- Each template has name, template string with `{variable}` placeholders, and description

### 2. Template persistence

- `load_user_templates()` reads from `~/.agent-loop/templates.json` at startup
- `save_user_templates()` writes non-builtin templates to JSON on `/template add`
- `dirs_config_path(filename)` helper resolves `$HOME/.agent-loop/<filename>`
- Creates directory if it doesn't exist
- `/template add` command now calls `save_user_templates()` after adding

### 3. Retry wired into API calls

- `RetryConfig::default()` initialized in both single-prompt and interactive paths
- `retry_with_backoff(&retry_config, ...)` wraps `backend.send_message()` calls
- Context snapshot via `ctx_snapshot: Vec<Message> = context.get_context().to_vec()` to avoid borrow conflicts in closures

### 4. Terminal spinner + rich display

- `Spinner` struct with `Arc<AtomicBool>` cross-thread signaling
- Background thread shows animated spinner (`| / - \`) with elapsed time
- `Spinner::start("Thinking...")` before API call, `spinner.stop()` after response
- `Drop` impl ensures cleanup even on panic
- Enhanced token display: `cost_tracker.format_summary() | context: ~N tokens (M msgs)`

### 5. Tests + verification

- 13 new assertions across 5 test predicates
- Fragment count updated: 27 → 28
- `cargo check` — 0 errors, 0 warnings
- Only expected files changed (agent_loop_module.pl, main.rs, test_agent_loop.pl, README.md)

## Test plan

- [x] 723 tests pass (13 new Phase 10 assertions), 1 pre-existing env-specific failure
- [x] `cargo check` — 0 errors, 0 warnings
- [x] md5sum baseline: only expected files changed
- [x] No Python or Prolog generated files touched
- [x] Rust fragment count updated (27 → 28)

## Metrics

| Metric | Phase 9 | Phase 10 |
|--------|---------|----------|
| Tests | 710 | 723 (+13) |
| rust_fragment/2 facts | 27 | 28 (+1: spinner) |
| Built-in templates | 6 | 16 (+10) |
| Template persistence | None | JSON save/load |
| Retry | Standalone | Wired into API calls |
| Spinner | None | Animated with elapsed time |

## Files changed

| File | Changes |
|------|---------|
| `agent_loop_module.pl` | Expanded template_manager fragment (16 built-ins + persistence), new spinner fragment, retry wiring in main_loop, enhanced token display, `/template add` saves |
| `test_agent_loop.pl` | 5 new test predicates (13 assertions), updated fragment count (28) |
| `README.md` | Updated test count (723), fragment count (28), Rust feature list, parity table (templates complete, spinner added) |
| `generated/rust/src/main.rs` | Auto-regenerated with all Phase 10 features |

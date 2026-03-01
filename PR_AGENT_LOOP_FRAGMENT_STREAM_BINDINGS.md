# PR: Fragment extraction + Prolog streaming + binding registry

**Branch**: `feat/agent-loop-fragment-stream-bindings`
**Base**: `main`

## Summary

Three workstreams increasing the data-to-code ratio and extending both Prolog runnability and cross-system integration.

### 1. Per-handler fragment split

Replaced the monolithic 327-line `py_fragment(agent_loop_command_handlers)` with 17 individual `py_fragment/2` facts, one per handler method:

| Fragment | Handler | Lines |
|----------|---------|-------|
| `handler_iterations_command` | `_handle_iterations_command` | 18 |
| `handler_backend_command` | `_handle_backend_command` | 27 |
| `handler_save_command` | `_handle_save_command` | 15 |
| `handler_load_command` | `_handle_load_command` | 20 |
| `handler_sessions_command` | `_handle_sessions_command` | 13 |
| `handler_format_command` | `_handle_format_command` | 15 |
| `handler_export_command` | `_handle_export_command` | 15 |
| `handler_cost_command` | `_handle_cost_command` | 15 |
| `handler_search_command` | `_handle_search_command` | 25 |
| `handler_stream_command` | `_handle_stream_command` | 9 |
| `handler_aliases_command` | `_handle_aliases_command` | 4 |
| `handler_templates_command` | `_handle_templates_command` | 4 |
| `handler_history_command` | `_handle_history_command` | 11 |
| `handler_undo_command` | `_handle_undo_command` | 6 |
| `handler_delete_command` | `_handle_delete_command` | 43 |
| `handler_edit_command` | `_handle_edit_command` | 34 |
| `handler_replay_command` | `_handle_replay_command` | 20 |

New predicates: `handler_fragment_name/2`, `emit_handler_fragments/1`, `cmd_has_handler/1`, `emit_handler_list/3`.

### 2. CLI override generation from facts

Extracted the 12-entry CLI argument override block from `agent_loop_main_body_pre_audit` into:
- 12 `cli_override/3` facts with 5 behavior types (`simple`, `set_true`, `clear_list`, `not_none_check`, `backend_special`)
- `generate_cli_overrides/1` + `emit_single_override/4` generators
- Fragment split: `agent_loop_main_body_pre_audit` -> `_pre_overrides` + generated + `_post_overrides` + `audit_levels_dict` + `_post_audit`

### 3. Prolog streaming support

Replaced the `/stream` stub with a working implementation:
- `streaming/1` dynamic fact (default `false`)
- `/stream` command toggles streaming on/off
- Streaming-aware LLM request path: checks `streaming_capable/1` before calling `send_request_streaming/4`
- `handle_response/4` suppresses content display when already streamed
- Ollama NDJSON streaming: `send_request_streaming_raw(api_local, ...)` with `http_open/3`, line-by-line JSON parsing via `json_read_dict/2`, immediate token printing
- `read_ndjson_stream/3` recursive stream reader with token accumulation

New fact: `streaming_capable(api_local)` — only Ollama (localhost HTTP) for now.

### 4. Binding registry integration

New file `agent_loop_bindings.pl` — optional overlay mapping agent-loop predicates to `binding/6`:
- 5 Python bindings: `tool_handler/2`, `slash_command/4`, `backend_factory/2`, `audit_profile_level/2`, `security_profile/2`
- 5 Prolog bindings: same predicates with direct fact access
- Effect declarations: pure/deterministic for lookups, effect(io) for backend creation
- `init_agent_loop_bindings/0` and `agent_loop_binding_summary/0`
- Reexports query API: `binding/6`, `bindings_for_target/2`, `is_pure_binding/2`, etc.

```
$ swipl -l agent_loop_bindings.pl -g "agent_loop_binding_summary, halt"
Agent-loop binding registry:
  Python bindings: 5
  Prolog bindings: 5
```

## New facts summary

| Fact | Count | Purpose |
|------|-------|---------|
| `cli_override/3` | 12 | CLI arg -> config field mapping with behavior |
| `streaming_capable/1` | 1 | Backend types supporting Prolog streaming |

## Verification

- Python zero-diff maintained
- Prolog loads cleanly: `swipl -l main.pl -g halt` -- no warnings
- Binding registration: 10 bindings (5 Python + 5 Prolog)
- Component registration: 33 instances across 3 categories (unchanged)
- Handler fragment count: 17
- 5 files changed/created

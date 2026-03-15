# feat(agent-loop): Phase 20 — Python parity lift + tool result caching, output parser, MCP support

## Summary

- **Python parity lift**: Port Gemini model validation, tool schema caching, and config reload fix from Rust to Python
- **Tool result caching**: TTL-based cache for non-destructive tool results in both Python and Rust targets
- **Structured output parsing**: Extract and validate JSON from model responses (fenced and bare) in both targets
- **MCP server support**: JSON-RPC 2.0 over stdio for tool discovery and execution in both targets

## Details

### Python Parity — Gemini Model Validation

| Aspect | Detail |
|--------|--------|
| Function | `validate_gemini_model(model, default)` + `extract_gemini_version(model)` |
| Rules | Flash >= 3.0, Pro >= 2.5, unknown models pass through |
| Wiring | Called in `create_backend_from_config()` for gemini backend factory |

### Python Parity — Tool Schema Caching

| Aspect | Detail |
|--------|--------|
| Function | `get_tool_schemas()` in `tools_generated.py` |
| Mechanism | Module-level `_TOOL_SCHEMAS_CACHE = None`, lazily built on first call |
| Benefit | Same list identity on repeated calls (no rebuild) |

### Python Parity — Config Reload Fix

| Aspect | Detail |
|--------|--------|
| Bug | `/reload` used `read_config_cascade()` (returns raw dict) + `getattr()` (fails silently) |
| Fix | Uses `load_config()` returning proper `AgentConfig`, compares all 7 fields |
| Fields | `backend`, `model`, `system_prompt`, `approval_mode`, `security_profile`, `stream`, `max_iterations` |

### Tool Result Caching (New — Both Targets)

| Aspect | Detail |
|--------|--------|
| Class | `ToolResultCache` in tools.py / tool_handler.rs |
| Key | `(tool_name, canonical_json(args))` |
| TTL | Default 60s, configurable via `tool_cache_ttl` in AgentConfig |
| Skip | Destructive tools (bash, write, edit) never cached |
| Methods | `get()`, `put()`, `clear()`, `size()` |

### Structured Output Parsing (New — Both Targets)

| Aspect | Detail |
|--------|--------|
| Module | `output_parser.py` / `output_parser.rs` |
| Classes | `OutputParser`, `ParsedOutput` |
| Extraction | Fenced code blocks (`\`\`\`json ... \`\`\``) first, then bare JSON objects |
| Validation | Optional `expected_keys` parameter reports missing top-level keys |

### MCP Server Support (New — Both Targets)

| Aspect | Detail |
|--------|--------|
| Module | `mcp_client.py` / `mcp_client.rs` |
| Transport | stdio (subprocess with stdin/stdout pipes) |
| Protocol | JSON-RPC 2.0 (`tools/list`, `tools/call`, `initialize`) |
| Classes | `MCPClient` (single server), `MCPManager` (multi-server) |
| Config | `mcp_servers` list in AgentConfig (`{name, command, args, env}`) |

## Test plan

- [x] 960 Prolog tests pass — 0 failures (+30 from Phase 19)
- [x] 108 cargo integration tests pass — 0 failures (+10 from Phase 19)
- [x] 92 Python integration tests pass — 0 failures (+23 from Phase 19)
- [x] 36 Prolog integration tests pass — 0 failures
- [x] `cargo check` — 0 errors

## Metrics

| Metric | Phase 19 | Phase 20 |
|--------|----------|----------|
| Prolog tests | 930 (0 fail) | 960 (0 fail) |
| Cargo tests | 98 | 108 |
| Python tests | 69 | 92 |
| Prolog integration | 36 | 36 |
| py_fragment count | 85 | 91 (+6) |
| rust_fragment count | 34 | 37 (+3) |
| New Prolog facts | — | cacheable_tool, tool_cache_default_ttl, mcp_transport, mcp_method (x3), mcp_jsonrpc_version |
| New config fields | — | tool_cache_ttl, mcp_servers |
| New generated modules | — | output_parser.py, output_parser.rs, mcp_client.py, mcp_client.rs |

## Files changed

| File | Changes |
|------|---------|
| `agent_loop_module.pl` | 6 new py_fragment, 3 new rust_fragment, 7 new facts, 2 new agent_config_field, Gemini validator wiring in factory, reload fix, new generators for output_parser + mcp_client |
| `test_agent_loop.pl` | 6 new test predicates (Phase 20), fragment count 34->37 |
| `test_integration.py` | 6 new test classes: TestOutputParser (6), TestToolResultCache (6), TestToolSchemaCache (2), TestGeminiValidation (5), TestMCPClient (4) |
| `README.md` | Updated test metrics, fragment counts, Rust feature list |
| Generated: `agent_loop.py` | validate_gemini_model + extract_gemini_version, fixed reload handler, factory wiring |
| Generated: `tools.py` | ToolResultCache class |
| Generated: `tools_generated.py` | get_tool_schemas() with lazy cache |
| Generated new: `output_parser.py` | OutputParser + ParsedOutput classes |
| Generated new: `mcp_client.py` | MCPClient + MCPManager classes |
| Generated: `rust/src/lib.rs` | pub mod output_parser, pub mod mcp_client |
| Generated new: `rust/src/output_parser.rs` | OutputParser struct |
| Generated new: `rust/src/mcp_client.rs` | McpClient + McpManager + McpServerConfig |
| Generated: `rust/src/tool_handler.rs` | ToolResultCache emission |
| Generated: `rust/src/types.rs` | tool_cache_ttl, mcp_servers fields |
| Generated: `rust/tests/integration_tests.rs` | 10 new Phase 20 tests |

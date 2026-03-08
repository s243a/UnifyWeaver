# PR: Composable generators — compiled security rules, export specs, profile field schema, fragment import derivation, unified config emitters

## Title

feat(agent-loop): Composable generators with declarative security rules, profile schema, fragment derivation

## Description

Makes generators more composable and target-polymorphic by replacing hardcoded `write` calls with declarative fact-driven emission. 6 generalizations across security, backends, config, and fragment infrastructure.

### Changes

#### 1. Fix `_BLOCKED_PREFIXES` name mismatch + `emit_security_check_predicates/2` (`agent_loop_components.pl` + `agent_loop_module.pl`)
- Fixed `compile_path_check_python` using `_BLOCKED_PATH_PREFIXES` → `_BLOCKED_PREFIXES`
- New `emit_security_check_predicates/2` — composed from `compile_path_check_rules` + `compile_command_check_rules` + thin wrappers (`check_path_allowed/2`, `check_command_allowed/2`, `set_security_profile/1`)
- Wired into `generate_prolog_security`, replacing **~25 hardcoded write calls** with 1 delegation
- Generated `security.pl` now exports `is_path_blocked/1` and `is_command_blocked/2` as composable building blocks

#### 2. `generator_export_specs(backends_init, ...)` (`agent_loop_components.pl` + `agent_loop_module.pl`)
- New `generator_export_specs(backends_init, Exports)` — derives `__all__` from `backend_factory/2` facts
- Excludes `optional_import(true)` backends (ClaudeAPIBackend, OpenAIBackend) automatically
- Wired into `generate_backends_init`, replacing split static/component-driven __all__ emission

#### 3. `security_profile_field/4` schema + `emit_security_profile_fields/2` (`agent_loop_module.pl` + `agent_loop_components.pl`)
- **20 `security_profile_field/4` facts** — declarative schema with `layer(N)`, `comment(Str)`, `inline_comment(Str)` annotations
- `emit_security_profile_fields/2` — iterates schema facts, groups by layer, emits Python dataclass fields with comments
- Wired into `generate_security_profiles`, replacing **~30 hardcoded write calls**
- Refactored `generate_profile_entry/3` — data-driven iteration over schema facts with type-dispatched value formatting (`emit_profile_field_if_present/4`, `emit_profile_field_value/4`), replacing **15 individual member/format branches**

#### 4. Fragment-driven import derivation (`agent_loop_components.pl`)
- `generator_fragments/2` — 3 facts mapping generators to their fragment names (tools, config, context)
- `derive_fragment_imports/2` — collects and deduplicates import specs from a generator's fragments
- `validate_generator_imports/2` — compares derived imports against declared `generator_import_specs`, returns `missing(Spec)` warnings

#### 5. Target-polymorphic `emit_config_section/3` (`agent_loop_components.pl`)
- `emit_config_section(S, api_key_env_vars, Options)` — Python: delegates to `emit_api_key_env_vars_py`; Prolog: emits `api_key_env_var/2` facts
- `emit_config_section(S, api_key_files, Options)` — same pattern for key files
- `emit_config_section(S, default_presets, Options)` — same pattern for presets

#### 6. Tests (`test_agent_loop.pl`)
- 10 new test predicates, 33 new assertions
- Tests cover all new predicates + roundtrip verification

### Tests

| Category | New Predicates | New Assertions |
|----------|---------------|----------------|
| emit_security_check_predicates | 1 | 5 |
| compiled rules variable name fix | 1 | 2 |
| generator_export_specs backends | 1 | 5 |
| security_profile_field facts | 1 | 3 |
| emit_security_profile_fields | 1 | 4 |
| generator_fragments | 1 | 3 |
| derive_fragment_imports | 1 | 3 |
| validate_generator_imports | 1 | 1 |
| emit_config_section | 1 | 4 |
| backends_init export roundtrip | 1 | 3 |
| **Total new** | **10** | **33** |

### Test results

| Suite | Count | Status |
|-------|-------|--------|
| Prolog unit | 324 | all pass |
| Prolog integration | 27 | all pass |
| Python integration | 59 | all pass |
| **Total** | **410** | **0 failures** |

### Files changed

| File | Changes | Summary |
|------|---------|---------|
| `agent_loop_components.pl` | +120/−1 | `emit_security_check_predicates/2`, `generator_export_specs(backends_init)`, `emit_security_profile_fields/2`, `generator_fragments/2`, `derive_fragment_imports/2`, `validate_generator_imports/2`, `emit_config_section/3`, fix `_BLOCKED_PREFIXES` |
| `agent_loop_module.pl` | +50/−65 | `security_profile_field/4` facts, wire compiled rules into Prolog generator, wire export specs into backends_init, wire profile fields, refactor `generate_profile_entry/3` |
| `test_agent_loop.pl` | +120/−0 | 10 new test predicates, 33 assertions |
| Generated files | regenerated | `security.pl` has `is_path_blocked/1`, `is_command_blocked/2`; `backends/__init__.py` uses declarative `__all__`; `security/profiles.py` fields from schema |

### Previous PRs

Builds on PR #748 (`feat/agent-loop-deeper-components`). Test count progression: 218 → 236 → 259 → 282 → 311 → 340 → 377 → **410**.

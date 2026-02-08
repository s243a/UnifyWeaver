# Unified config with uwsal.json and API key resolution

## Summary

- **uwsal.json** (Unify Weaver, Simple Agent Loop) is the primary config file, checked before coro.json
- **API key resolution** follows a 5-level priority chain: CLI arg > env var > config file > coro.json fallback > standard file locations
- **`--no-fallback`** extended to skip coro.json for both command resolution and config discovery (uwsal.json is always checked)

## API key resolution priority

```
1. --api-key CLI argument
2. Backend-specific env var (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, etc.)
3. uwsal.json (keys.<backend> or top-level api_key)
4. coro.json fallback (skipped with --no-fallback)
5. Standard file locations (~/.anthropic/api_key, etc.)
```

## uwsal.json format

```json
{
  "api_key": "sk-or-...",
  "model": "moonshotai/kimi-k2.5",
  "base_url": "https://openrouter.ai/api/v1",
  "keys": {
    "openrouter": "sk-or-...",
    "anthropic": "sk-ant-...",
    "openai": "sk-..."
  }
}
```

- Top-level `api_key` / `model` / `base_url` — default for any backend
- `keys` object — per-provider keys, looked up by backend type
- Both are optional; backends use what they find

## Config file search order

```
1. CWD/uwsal.json
2. ~/uwsal.json
3. CWD/coro.json     (skipped with --no-fallback)
4. ~/coro.json       (skipped with --no-fallback)
```

## Files changed

| File | Change |
|------|--------|
| `config.py` | Added `read_config_cascade()` and `resolve_api_key()` |
| `agent_loop.py` | Backend factory uses shared config; extended `--no-fallback` |
| `openrouter_api.py` | Removed `_read_coro_config()`; simplified constructor |
| `coro.py` | `_find_config()` and `_read_coro_config()` check uwsal.json first |

## Test plan

- [x] Default — reads API key from coro.json fallback
- [x] `--api-key` — CLI key takes precedence (401 with fake key confirms it was used)
- [x] `--no-fallback` without uwsal.json — correctly fails with config error
- [x] `--no-fallback` with ~/uwsal.json — works, reads from uwsal.json
- [ ] Per-provider keys via `keys` object in uwsal.json
- [ ] Env var precedence over config file

---

Generated with [Claude Code](https://claude.com/claude-code)

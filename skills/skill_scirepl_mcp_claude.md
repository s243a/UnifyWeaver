# sciREPL MCP — Claude Code Workflow

Use this skill when automating the sciREPL app through its MCP server (`scirepl_*` tools).

## Tool Inventory

The SciREPL MCP exposes tools prefixed `mcp__scirepl__scirepl_*`. Key ones:

| Tool | Purpose |
|------|---------|
| `scirepl_connect` | Launch browser, navigate to sciREPL, accept privacy |
| `scirepl_disconnect` | Tear down the browser session |
| `scirepl_get_visible_state` | Inspect modals, cells, buttons, console logs |
| `scirepl_screenshot` | Full-page screenshot |
| `scirepl_execute_code` | Run arbitrary code in any kernel |
| `scirepl_run_all_cells` | Programmatic run-all via JS API |
| `scirepl_run_all_cells_ui` | Visible hamburger-menu "Run All Cells" |
| `scirepl_get_cells` | List cells with metadata |
| `scirepl_create_cell` / `scirepl_delete_cell` | Add/remove cells |
| `scirepl_set_cell_language` | Change cell kernel |
| `scirepl_install_package` | Install from catalog |
| `scirepl_list_catalog` | List available packages |
| `scirepl_import_file` / `scirepl_export_workbook` | Import/export notebooks |
| `scirepl_get_kernel_status` | Check kernel readiness |
| `scirepl_get_settings` / `scirepl_set_setting` | Read/write localStorage settings |
| `scirepl_get_cell_outputs_detailed` | Model + DOM output with `Out [n]` labels |
| `scirepl_get_browser_debug_info` | CDP URL, storage, service workers, console |
| `scirepl_install_local_package` | Install a package from a local zip, bypassing GitHub |
| `scirepl_vfs_write` | Write a single local file into the Prolog or Shared VFS |
| `scirepl_vfs_overlay_dir` | Overlay a local directory onto the VFS recursively |
| `scirepl_vfs_list` | List files in the Prolog Emscripten VFS |

## Coordination with Playwright MCP

Claude Code also has a general Playwright MCP (`mcp__playwright__browser_*`). These are **independent browser instances**. Rules:

1. **Do not cross sessions.** Once `scirepl_connect` starts a session, complete the entire sciREPL workflow using `scirepl_*` tools. Do not use `mcp__playwright__browser_navigate` to open the same sciREPL URL in a second browser — that causes "which browser?" confusion.

2. **Playwright MCP is fine for other sites.** If the task involves both sciREPL and an unrelated web page, use `scirepl_*` for sciREPL and `mcp__playwright__*` for the other page. They won't interfere.

3. **Shared-session debug mode** is the one exception. When debugging browser-level issues (cache, service workers, storage drift), connect with `scirepl_connect(debugMode: true, remoteDebuggingPort: 9223)`. This exposes the sciREPL browser on CDP port 9223. If needed, Playwright MCP can then `browser_navigate` to `http://127.0.0.1:9223` for deeper inspection of the *same* session — but clearly distinguish observations from each surface.

## Standard Workflow

```
1. scirepl_connect
2. scirepl_install_package  (if loading a workbook)
3. scirepl_run_all_cells_ui  (visible) or scirepl_run_all_cells (programmatic)
4. scirepl_screenshot + scirepl_get_cell_outputs_detailed
5. scirepl_disconnect  (when done)
```

## Choosing Run Method

- **`scirepl_run_all_cells_ui`** — default for any task involving visible output inspection, warning triage, or when the user mentions "run from menu" / "use the UI". This opens the hamburger menu and clicks Run All Cells in the actual browser.
- **`scirepl_run_all_cells`** — for batch/programmatic execution where visual confirmation isn't needed.

## Diagnosing Warnings

When the task is about warnings visible in notebook output:

1. Use `scirepl_run_all_cells_ui` — the warning must be reproduced through the visible UI path.
2. Read `scirepl_get_cell_outputs_detailed` first — this shows both model-backed and DOM-visible output, including `Out [n]` labels.
3. Take a `scirepl_screenshot` for visual confirmation.
4. Console logs (`scirepl_get_visible_state`) are supporting evidence only — never diagnose from console alone.

If the user can see a warning in a cell but tooling reports none, treat it as an instrumentation gap in the MCP server. Investigate the DOM extraction logic in `scirepl-mcp-server.js` before concluding the warning doesn't exist.

## Browser Debug Mode

For issues involving cache, stale assets, service workers, or storage state:

```
1. scirepl_connect(debugMode: true, remoteDebuggingPort: 9223)
2. scirepl_get_browser_debug_info  → note CDP URL
3. Continue normal workflow in the same session
4. If deeper inspection needed, attach auxiliary tooling to the reported CDP URL
```

Keep observations from each surface (MCP-owned browser vs auxiliary DevTools) clearly separated.

## Supported Languages

Python (Pyodide), R (webR), SWI-Prolog, Bash, JavaScript, Lua.

## Key Settings

| Setting | Values | Effect |
|---------|--------|--------|
| `scirepl_privacy_accepted` | `1`/`0` | Privacy policy |
| `scirepl_auto_download` | `1`/`0` | Auto-download runtimes |
| `scirepl_auto_switch_workbook` | `1`/`0` | Auto-switch on install |
| `scirepl_auto_execute` | `1`/`0` | Auto-execute on import |

## Local Development Workflow

When iterating on UnifyWeaver source files and testing changes in SciREPL without publishing a new GitHub release:

### Option 1: Dev packages env var (automatic)

Set `SCIREPL_DEV_PACKAGES` to a directory containing local `.zip` packages. When `scirepl_install_package` is called, it checks this directory first (matching by name) and installs from local instead of the catalog.

Already configured in `.mcp.json` to point at `prototype/www/packages/`.

### Option 2: VFS overlay (surgical)

After a normal package install, overwrite individual files:
```
1. scirepl_connect
2. scirepl_install_package(name: "UnifyWeaver SciREPL")
3. scirepl_vfs_write(localPath: "/path/to/fixed/component_registry.pl",
                     vfsPath: "/user/unifyweaver/core/component_registry.pl")
4. Run cells as normal
```

Or overlay an entire directory:
```
scirepl_vfs_overlay_dir(localDir: "/path/to/src/unifyweaver",
                        vfsBasePath: "/user/unifyweaver",
                        glob: "**/*.pl")
```

### Option 3: Local package install (explicit)

```
scirepl_install_local_package(zipPath: "/path/to/unifyweaver_scirepl.zip")
```

### Rebuilding the local package zip

To update a single file in the package zip:
```bash
cd /path/to/UnifyWeaver
zip -u examples/sci-repl/prototype/www/packages/unifyweaver_scirepl.zip src/unifyweaver/core/component_registry.pl
```

## Server Location

- Implementation: `examples/sci-repl/scirepl-mcp-server.js`
- Dependencies: `examples/sci-repl/package.json`
- Original Cline skill: `skills/skill_scirepl_mcp.md`
- Docs: `examples/sci-repl/MCP_README.md`
- Local package zip: `examples/sci-repl/prototype/www/packages/unifyweaver_scirepl.zip`

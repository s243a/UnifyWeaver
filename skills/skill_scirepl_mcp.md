# sciREPL MCP Single-Browser Workflow

Use this skill when automating the sciREPL app through its custom MCP server.

## Primary Objective for Warning Triage

When the user asks about **warnings in workbook output**, the goal is to inspect the **rendered cell output first**.

This means Kimi must:

1. use the built-in sciREPL MCP server,
2. actually trigger **Run All Cells** from the visible hamburger menu,
3. look at the **cell output warnings**,
4. treat console warnings/errors as secondary evidence only.

## Goal

Avoid browser confusion by ensuring package installation, hamburger-menu interaction, Run All Cells, screenshots, and output inspection all happen in the **same browser session** owned by the sciREPL MCP server.

## Core Rule

For one sciREPL session, do **not** split work across:

- the sciREPL MCP server's Playwright browser, and
- Chrome DevTools MCP / a separate browser tab

If the workflow starts with `scirepl_connect`, continue using sciREPL MCP tools for the rest of that notebook session.

Do **not** switch to Chrome DevTools MCP to click the hamburger or inspect the notebook once the sciREPL MCP session has started.

## Debugging the sciREPL MCP Server Itself

The strict single-browser rule is for **user/workbook troubleshooting**.

When the task changes to **debugging the sciREPL MCP server implementation itself**, treat that as a separate mode:

- **Normal triage mode:** use sciREPL MCP end-to-end in one browser session.
- **Server-debug mode:** deeper inspection is allowed, but the goal is to improve sciREPL MCP observability or verify what its browser session is exposing.

In server-debug mode, prefer expanding sciREPL MCP observability first:

- add richer output-inspection tools
- inspect DOM-visible rendered output, not only `_cells[].outputHtml`
- capture `Out [n]` labels and rendered text blocks
- when needed, use an explicit shared-session browser-debug mode so the sciREPL MCP browser can also be inspected through a DevTools/CDP surface

If a second debugging surface is used, explicitly distinguish:

- what was observed in the sciREPL MCP-owned browser
- what was observed in any auxiliary debugging browser/tool

Do not collapse those into one source of truth.

### Shared-Session Browser-Debug Mode

When the suspected bug may involve browser/runtime state rather than notebook semantics alone — for example:

- cache clearing issues
- stale JS/CSS/assets after reload
- service worker interference
- web worker / shared worker lifecycle issues
- storage persistence or invalid cached state

— prefer an explicit shared-session debug path:

1. connect sciREPL MCP with browser-debug mode enabled
2. capture `scirepl_get_browser_debug_info`
3. continue the notebook workflow in that same MCP-controlled session
4. if needed, attach auxiliary DevTools tooling to the reported browser URL

This is different from ad hoc cross-browser mixing: the goal is to inspect the **same browser session intentionally**, not two unrelated ones.

## Preferred Tool Sequence

1. `scirepl_connect`
2. `scirepl_install_package` or `scirepl_open_catalog`
3. `scirepl_run_all_cells_ui`
4. `scirepl_screenshot` to capture rendered outputs
5. `scirepl_get_visible_state` to inspect the visible notebook state

If the task is specifically about warnings in notebook output, this sequence is mandatory.

## Choose the Right Run Tool

- Use `scirepl_run_all_cells_ui` when the user says things like:
  - “use the buttons on the UI”
  - “open the hamburger icon”
  - “I want to see the output cells show”
  - “run all cells from the menu”
  - “look at the warning in the cell output”
  - “inspect the actual output, not the console”

- Use `scirepl_run_all_cells` only when programmatic execution is acceptable.

For workbook-warning troubleshooting, `scirepl_run_all_cells_ui` is the default.

## Evidence Priority

When diagnosing warnings:

1. **First priority:** rendered cell output text
2. **Second priority:** screenshot of the rendered notebook
3. **Third priority:** visible UI state from `scirepl_get_visible_state`
4. **Last priority:** console logs/errors

For difficult cases, “rendered cell output text” must include both:

- output derived from the cell model, and
- DOM-visible output text actually shown in the notebook UI (including `Out [n]` cards when present)

Kimi must not conclude that a warning has been diagnosed from console logs alone if the task is about notebook output warnings.

## Troubleshooting

If Kimi appears to click the hamburger in the wrong browser:

1. Stop mixing MCP servers for that session
2. Reconnect using `scirepl_connect`
3. Re-run the flow entirely with sciREPL MCP tools
4. Use `scirepl_get_visible_state` to confirm which UI is controlled

If Kimi inspects console logs before cell outputs:

1. Re-run using `scirepl_run_all_cells_ui`
2. Capture a screenshot after execution
3. Read the visible cell output warning text first
4. Only then use console logs to support, not replace, the diagnosis

If the user can visibly read warning text in a cell such as `Out [47]`, but tooling reports “0 warnings found”, assume an **instrumentation gap** first:

1. inspect DOM-visible output for that cell
2. inspect model-backed output fields separately
3. improve the sciREPL MCP extractor/tooling
4. only then conclude whether the warning is absent or present

## Forbidden Shortcuts

Do not:

- replace `scirepl_run_all_cells_ui` with programmatic per-cell execution when the user asked for the visible menu flow
- diagnose output warnings from console logs alone
- claim the warning was reproduced unless the visible **Run All Cells** action was actually triggered in the sciREPL MCP browser

## Expected Outcome

After the fix, Kimi should be able to:

- install “UnifyWeaver SciREPL” in the sciREPL MCP browser
- open the hamburger menu in that same browser
- trigger **Run All Cells** visibly via the UI
- inspect output cells and warnings without cross-browser confusion
- prioritize notebook output warnings over console diagnostics
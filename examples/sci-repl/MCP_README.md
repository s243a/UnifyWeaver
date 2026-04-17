# sciREPL MCP Server

An MCP (Model Context Protocol) server that provides automated tools for sciREPL integration. This server supports both programmatic control and visible UI-driven control of sciREPL notebooks, including cell execution, package management, and workbook operations.

## Overview

sciREPL is a multi-language scientific REPL (Read-Eval-Print Loop) supporting Python (Pyodide), R (webR), Prolog (SWI), Bash, JavaScript, and Lua. This MCP server automates common workflows:

- **Connection Management**: Auto-accept privacy policies, skip download confirmations
- **Cell Execution**: Execute individual cells, run all cells programmatically, or trigger Run All Cells through the visible UI menu
- **Package Management**: Browse and install packages/workbooks from the catalog
- **File Operations**: Import/export workbooks, browse shared filesystem
- **Settings**: Manage sciREPL settings programmatically

## Important: Use a Single Browser Session

This MCP server launches and owns its own Playwright browser session. To avoid browser confusion:

- Use the **sciREPL MCP server end-to-end** for a sciREPL workflow
- Do **not** mix sciREPL MCP browser actions with Chrome DevTools MCP actions for the same notebook session
- If you install a package with `scirepl_install_package`, then open the hamburger menu and trigger `Run All Cells`, those steps must all happen through this same MCP server session

This is especially important when the goal is **visible UI execution** where the user wants to see the hamburger menu open and output cells render on screen.

### Important Distinction: Normal Triage vs MCP Server Debugging

The single-browser rule is the default for **notebook/workbook troubleshooting**.

However, debugging the **sciREPL MCP server itself** is a different task. In that case:

- prefer enriching the sciREPL MCP server so it exposes more of what the notebook is visibly rendering
- treat DOM-visible notebook output as first-class evidence, not just `_cells[].outputHtml`
- if auxiliary tooling is used for deeper investigation, clearly distinguish it from the sciREPL MCP-owned browser session

The goal is to avoid cross-browser confusion during normal use while still allowing the sciREPL MCP server to become easier to debug and validate.

## Installation

### Prerequisites

- Node.js 18+ installed
- sciREPL server running (typically on http://localhost:8085)

### Setup

1. Navigate to the sciREPL MCP directory:
```bash
cd examples/sci-repl
```

2. Install dependencies:
```bash
npm install
```

3. The MCP server is automatically configured via `cline_mcp_settings.json`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCIREPL_URL` | `http://localhost:8085` | URL of the sciREPL instance |
| `SCIREPL_HEADLESS` | `true` | Run browser in headless mode |
| `SCIREPL_TIMEOUT` | `120000` | Timeout for operations (ms) |
| `SCIREPL_DEBUG_MODE` | `false` | Launch Chromium with a remote debugging port for shared-session browser debugging |
| `SCIREPL_REMOTE_DEBUGGING_PORT` | `9223` | Remote debugging port exposed when debug mode is enabled |
| `SCIREPL_BROWSER_URL` | `` | Optional CDP browser URL to attach the sciREPL MCP server to an already-debuggable browser |

### MCP Settings

The server is configured in `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "scirepl": {
      "disabled": false,
      "timeout": 120,
      "type": "stdio",
      "command": "node",
      "args": [
        "c:/Users/johnc/Dropbox/projects/UnifyWeaver/examples/sci-repl/scirepl-mcp-server.js"
      ],
      "env": {
        "SCIREPL_URL": "http://localhost:8085",
        "SCIREPL_HEADLESS": "true",
        "SCIREPL_TIMEOUT": "120000"
      }
    }
  }
}
```

## Available Tools

### Connection & Navigation

| Tool | Description |
|------|-------------|
| `scirepl_connect` | Connect to sciREPL instance, auto-accept privacy, wait for Pyodide |
| `scirepl_disconnect` | Clean up browser resources |
| `scirepl_open_menu` | Open the main menu |
| `scirepl_close_modal` | Close any open modal/dialog |
| `scirepl_get_visible_state` | Inspect visible modals, cells, buttons, and recent logs from the MCP-controlled browser |
| `scirepl_get_browser_debug_info` | Inspect shared-session debugging details such as the CDP browser URL, page URL, storage keys, service workers, and recent console logs |
| `scirepl_get_cell_outputs_detailed` | Inspect notebook outputs using both model data and DOM-visible rendered output, including `Out [n]` labels when present |
| `scirepl_screenshot` | Take a screenshot of the page |

### Cell Operations

| Tool | Description |
|------|-------------|
| `scirepl_execute_code` | Execute arbitrary code in any language kernel |
| `scirepl_execute_cell` | Execute a specific cell by index |
| `scirepl_run_all_cells` | Run all cells programmatically in the workbook (optional stop-on-error) |
| `scirepl_run_all_cells_ui` | Open the visible UI menu and click **Run All Cells** in the MCP-controlled browser |
| `scirepl_get_cells` | List all cells with metadata |
| `scirepl_create_cell` | Add a new cell |
| `scirepl_delete_cell` | Remove a cell by index |
| `scirepl_set_cell_language` | Change language of a cell |

### Package & Workbook Management

| Tool | Description |
|------|-------------|
| `scirepl_open_catalog` | Open Browse Packages & Workbooks |
| `scirepl_list_catalog` | List all available packages |
| `scirepl_install_package` | Install a package by name |
| `scirepl_import_file` | Import .srwb, .ipynb, or .py files |
| `scirepl_export_workbook` | Export to .srwb, .ipynb, or package |

### Settings & Status

| Tool | Description |
|------|-------------|
| `scirepl_get_settings` | Get all sciREPL settings |
| `scirepl_set_setting` | Set a setting value |
| `scirepl_get_kernel_status` | Check status of all kernels |
| `scirepl_accept_privacy` | Accept privacy policy |
| `scirepl_get_shared_files` | List files in SharedVFS |

### Utility

| Tool | Description |
|------|-------------|
| `scirepl_wait_for` | Wait for text to appear on page |

## Usage Examples

### Recommended Single-Browser UI Workflow

```
1. Connect to sciREPL
   → scirepl_connect

2. Install the workbook/package in the same browser session
   → scirepl_install_package
   name: "UnifyWeaver SciREPL"
   waitForComplete: true

3. Inspect visible UI state if needed
   → scirepl_get_visible_state

4. Trigger the visible menu action
   → scirepl_run_all_cells_ui
   waitForOutput: true

5. Capture evidence from the same browser session
   → scirepl_screenshot
   → scirepl_get_cell_outputs_detailed
```

### Shared-Session Browser-Debug Workflow

Use this when the issue may involve browser internals such as cache state, service workers, workers, storage, stale assets, or other runtime/debugging concerns.

```
1. Connect in browser-debug mode
   → scirepl_connect
   debugMode: true
   remoteDebuggingPort: 9223

2. Confirm the MCP-controlled session's debug details
   → scirepl_get_browser_debug_info

3. Continue normal sciREPL workflow in that same session
   → scirepl_install_package
   → scirepl_run_all_cells_ui
   → scirepl_get_cell_outputs_detailed

4. Attach Chrome DevTools tooling to the same browser URL
   browser URL: http://127.0.0.1:9223
```

This mode is for **server-debugging / browser-state investigation**. For ordinary workbook triage, keep using the sciREPL MCP workflow end-to-end without mixing tools.

### Basic Workflow

```
1. Connect to sciREPL
   → scirepl_connect

2. Check kernel status
   → scirepl_get_kernel_status

3. Execute Python code
   → scirepl_execute_code
   code: "import numpy as np; np.linspace(0, 10, 5)"
   language: "python"

4. Run all cells in workbook programmatically
   → scirepl_run_all_cells
   stopOnError: true
```

### Visible UI Execution

Use this when the user explicitly wants to see the hamburger menu and output cells update visually:

```
1. Connect to sciREPL
   → scirepl_connect

2. Open the hamburger menu
   → scirepl_open_menu

3. Click the visible Run All Cells menu item
   → scirepl_run_all_cells_ui
   waitForOutput: true
```

### Package Installation

```
1. List available packages
   → scirepl_list_catalog

2. Install a package
   → scirepl_install_package
   name: "Life Expectancy Analysis"
   waitForComplete: true
```

### Workbook Execution

```
1. Import a workbook
   → scirepl_import_file
   filePath: "./my_notebook.ipynb"

2. Get cell information
   → scirepl_get_cells

3. Run all cells
   → scirepl_run_all_cells_ui   # for visible UI execution
   or
   → scirepl_run_all_cells      # for programmatic execution

4. Export results
   → scirepl_export_workbook
   format: "ipynb"
   outputPath: "./results.ipynb"
```

## Key Features

### Automatic Modal Handling

The server automatically:
- Accepts privacy policy on connect
- Skips download confirmation dialogs (via `scirepl_auto_download` setting)
- Auto-accepts any browser dialogs
- Keeps UI actions and package installs inside the same Playwright browser session

### Shared-Session Browser Debugging

The MCP server now supports an explicit browser-debug mode for cases where the problem may involve browser/runtime state rather than only notebook logic.

- `scirepl_connect` can now:
  - launch Chromium with `--remote-debugging-port=<port>` when `debugMode: true`, or
  - attach to an already-debuggable browser via `browserUrl`
- `scirepl_get_browser_debug_info` reports:
  - whether browser-debug mode is active
  - the CDP browser URL to inspect
  - the current page URL/title
  - local/session storage keys
  - service worker registrations
  - recent console logs

This is intended for investigating issues such as:

- cache clearing behavior
- stale assets after reloads
- service worker interference
- worker/runtime lifecycle issues
- storage/state drift across runs

### UI vs Programmatic Execution

These two tools are intentionally different:

- `scirepl_run_all_cells` = **programmatic execution** via JavaScript APIs
- `scirepl_run_all_cells_ui` = **visible UI execution** via the hamburger menu in the MCP-controlled browser

If the user wants to visually confirm output cells rendering, prefer `scirepl_run_all_cells_ui`.

### Multi-Language Support

Supported languages for code execution:
- `python` - Python 3 via Pyodide
- `r` - R via webR
- `prolog` - SWI-Prolog
- `bash` - Brush WASM shell
- `javascript` - Browser JavaScript
- `lua` - Lua via Fengari

### Settings Management

Key settings accessible via `scirepl_set_setting`:

| Setting | Values | Description |
|---------|--------|-------------|
| `scirepl_privacy_accepted` | `1`/`0` | Privacy policy acceptance |
| `scirepl_auto_download` | `1`/`0` | Auto-download runtimes |
| `scirepl_auto_switch_workbook` | `1`/`0` | Auto-switch on workbook install |
| `scirepl_auto_execute` | `1`/`0` | Auto-execute on import |

## Troubleshooting

### Connection Issues

1. Verify sciREPL is running:
   ```bash
   curl http://localhost:8085
   ```

2. Check environment variables:
   ```bash
   echo $SCIREPL_URL
   ```

3. Try connecting with `waitForPyodide: false` first

### Runtime Download Issues

If Pyodide/R/Prolog won't download:
1. Check internet connection
2. Manually accept privacy policy once via UI
3. Set `scirepl_auto_download: "1"` in settings

### Cell Execution Errors

- Check kernel status with `scirepl_get_kernel_status`
- Verify the language kernel is ready
- Check browser console logs

### Browser Confusion / Wrong Window Issues

If actions appear to happen in the wrong browser window:

1. Do not mix sciREPL MCP and Chrome DevTools MCP for the same sciREPL session
2. Reconnect with `scirepl_connect`
3. Perform install, menu open, run-all, screenshot, and state inspection using only sciREPL MCP tools
4. Use `scirepl_get_visible_state` to confirm what the MCP-controlled browser is showing

## Architecture

### Components

```
┌─────────────────┐      ┌──────────────┐      ┌────────────────┐
│  MCP Client     │──────│  MCP Server  │──────│  Playwright    │
│  (Cline/Claude) │      │  (this code) │      │  (Browser)     │
└─────────────────┘      └──────────────┘      └────────────────┘
                                                        │
                                              ┌─────────▼─────────┐
                                              │  sciREPL          │
                                              │  (localhost:8085) │
                                              └───────────────────┘
```

### Data Flow

1. MCP Client sends tool call request
2. MCP Server receives and validates request
3. Playwright executes browser automation
4. sciREPL JavaScript API is invoked
5. Results returned to MCP Client

## Related Files

- `scirepl-mcp-server.js` - Main MCP server implementation
- `prototype/server.js` - sciREPL development server
- `prototype/www/js/kernel_manager.js` - sciREPL kernel management
- `prototype/www/js/package_catalog.js` - Package catalog functionality

## License

MIT License - See main UnifyWeaver repository

## Version History

- **1.0.0** - Initial release
  - 25+ MCP tools for sciREPL automation
  - Multi-language kernel support
  - Package catalog integration
  - Automatic modal/dialog handling

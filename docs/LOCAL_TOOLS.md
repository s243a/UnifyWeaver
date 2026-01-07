# Local Tools Directory (.local/)

The `.local/` directory provides a space for tools and subprojects that extend UnifyWeaver's functionality but aren't distributed with the main repository.

## Relationship to Main Project

The main project may define **interfaces** that invoke tools in `.local/`, but the **implementations** live separately:

- Main project scripts can call `.local/bin/` wrappers
- The main project doesn't provide the actual tool implementations
- Users set up their own `.local/` tools based on their needs

This separation allows:
- Optional features without adding dependencies to the main project
- User-specific tool configurations
- Independent versioning of tool implementations

## Why .local/?

Code in `.local/` is kept separate from the main project for various reasons:

- **Private/proprietary code**: Tools that won't be open-sourced or released publicly
- **External dependencies**: Browser automation (Playwright, Chrome), ML models, large datasets
- **User-specific configs**: API keys, account credentials, local paths
- **Experimental tools**: Scripts under development that may not be ready for distribution
- **Separate version control**: Subprojects with their own git history and licensing

## Directory Structure

```
.local/
├── bin/           # Wrapper scripts (add to PATH for convenience)
├── lib/           # Shared libraries and utilities
└── tools/         # Subprojects (can be separate git repos)
    └── browser-automation/   # Example: Pearltrees scraping tools
        ├── .git/
        ├── README.md
        ├── docs/
        └── scripts/
```

## Usage

### Adding to PATH

```bash
export PATH="$PATH:$(pwd)/.local/bin"
```

### Creating a New Tool Subproject

```bash
mkdir -p .local/tools/my-tool
cd .local/tools/my-tool
git init
```

### Example: Browser Automation

The `browser-automation` subproject provides tools for fetching Pearltrees data via the API when exports are incomplete:

1. Uses Playwright MCP for browser control
2. Fetches `getTreeAndPearls` API with authenticated session
3. Parses response into repair entries (PagePearls, Trees, RefPearls)
4. Outputs JSONL compatible with the repair pipeline

## Notes

- The `.local/` directory is gitignored and won't be committed
- Each subproject in `tools/` can have its own git repo
- Wrapper scripts in `bin/` provide a clean interface to tools
- Documentation for specific tools lives within their subproject

# SciREPL

A mobile-first scientific Python REPL powered by Pyodide.

## Versions

- **[prototype/](prototype/)** — Hand-crafted reference implementation (current version)
  - Standalone repo: [s243a/SciREPL](https://github.com/s243a/SciREPL)
  - Maintained here as git submodule
- **[generated/](generated/)** — UnifyWeaver-generated version (coming soon)

## Download

**Latest APK:** https://github.com/s243a/SciREPL/releases

## For Developers

```bash
# Clone with submodules
git clone --recursive https://github.com/s243a/UnifyWeaver.git

# Or if already cloned
git submodule update --init --recursive
```

## Quick Start

```bash
cd examples/sci-repl/prototype
python3 -m http.server 8080 -d www
```

Open http://localhost:8080

See [prototype/README.md](prototype/README.md) for full documentation.

## Vision

The prototype demonstrates the full feature set. The `generated/` version will show how UnifyWeaver can generate the same app from high-level specifications, demonstrating:

- Prolog → TypeScript/JavaScript compilation
- Declarative UI generation
- Python bridge code generation
- Cross-target compilation (web + mobile)

## License

MIT — see prototype/LICENSE

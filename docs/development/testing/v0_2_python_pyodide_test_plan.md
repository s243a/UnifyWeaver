# Python Pyodide Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Pyodide (WebAssembly Python) code generation target testing

## Overview

This test plan covers the Python Pyodide target for UnifyWeaver, which generates Python code optimized for running in web browsers via Pyodide (Python compiled to WebAssembly). This enables client-side Datalog query execution without a server.

## Prerequisites

### System Requirements

- Node.js 18+ (for testing)
- Python 3.11+ (for local validation)
- Modern web browser (Chrome, Firefox, Safari)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Node.js (for test runner)
node --version

# Verify Python
python3 --version

# Verify Prolog
swipl --version

# Test Pyodide loading (requires network)
node -e "
const { loadPyodide } = require('pyodide');
loadPyodide().then(py => console.log('Pyodide version:', py.version));
"
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_python_pyodide_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `async_main` | async def main() | Async entry |
| `no_threads` | No threading | Single-threaded |
| `web_apis` | js module usage | JS interop |
| `pure_python` | No C extensions | Compatible code |

#### 1.2 Web Integration

```bash
swipl -g "use_module('tests/core/test_pyodide_web'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `dom_access` | js.document | DOM access |
| `fetch_api` | pyfetch() | HTTP requests |
| `event_handlers` | create_proxy() | Event binding |
| `local_storage` | js.localStorage | Storage access |

### 2. Compilation Tests

#### 2.1 Pyodide Execution (Node.js)

```bash
./tests/integration/test_pyodide_node.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `load_pyodide` | Initialize runtime | Pyodide loads |
| `run_python` | runPython() | Code executes |
| `micropip` | micropip.install | Packages install |
| `result_conversion` | toJs() | Values convert |

#### 2.2 Browser Tests

```bash
./tests/integration/test_pyodide_browser.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `chrome_execution` | Run in Chrome | Works |
| `firefox_execution` | Run in Firefox | Works |
| `safari_execution` | Run in Safari | Works |
| `service_worker` | SW context | Works |

### 3. Generated Code Structure

```python
# Pyodide-compatible generated query
# No threading, no C extensions, async-friendly

from js import console, document, JSON
from pyodide.ffi import to_js, create_proxy

class Fact:
    __slots__ = ('relation', 'args')

    def __init__(self, relation: str, args: tuple):
        self.relation = relation
        self.args = args

    def __hash__(self):
        return hash((self.relation, self.args))

    def __eq__(self, other):
        return self.relation == other.relation and self.args == other.args

    def to_dict(self):
        return {"relation": self.relation, "args": list(self.args)}

class GeneratedQuery:
    def __init__(self):
        self.facts: set = set()
        self.delta: set = set()

    def init_facts(self):
        self.facts.add(Fact("parent", ("john", "mary")))
        self.facts.add(Fact("parent", ("mary", "susan")))
        self.delta = set(self.facts)

    def apply_rules(self):
        new_facts = set()
        for f in self.delta:
            if f.relation == "ancestor":
                for g in self.facts:
                    if g.relation == "parent" and f.args[1] == g.args[0]:
                        new_facts.add(Fact("ancestor", (f.args[0], g.args[1])))
        return new_facts

    def solve(self):
        self.init_facts()
        while self.delta:
            new_facts = self.apply_rules()
            self.delta = new_facts - self.facts
            self.facts |= self.delta
        return self.facts

    def to_json(self):
        return [f.to_dict() for f in self.facts]

# Entry point for browser
async def run_query():
    query = GeneratedQuery()
    results = query.solve()
    return to_js([f.to_dict() for f in results])

# Export for JavaScript
query_runner = create_proxy(run_query)
```

### 4. HTML Integration Tests

#### 4.1 Basic HTML Page

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
</head>
<body>
    <div id="results"></div>
    <script>
        async function main() {
            const pyodide = await loadPyodide();
            await pyodide.loadPackage('micropip');

            // Load generated query
            await pyodide.runPythonAsync(`
                ${generatedPythonCode}
            `);

            // Run query
            const results = await pyodide.runPythonAsync('await run_query()');
            document.getElementById('results').textContent = JSON.stringify(results.toJs());
        }
        main();
    </script>
</body>
</html>
```

### 5. Performance Tests

#### 5.1 Browser Performance

```bash
./tests/perf/test_pyodide_performance.sh
```

**Benchmarks**:
| Test | Expected Time | Notes |
|------|---------------|-------|
| Simple query | < 100ms | After warmup |
| 1000 facts | < 500ms | Memory limited |
| First load | < 5s | WASM initialization |

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/python_pyodide_target'),
    compile_to_pyodide(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Node.js Test

```bash
# Run with Pyodide in Node.js
node tests/pyodide/run_query.js
```

### Full Test Suite

```bash
./tests/run_pyodide_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYODIDE_VERSION` | Pyodide version | `0.24.1` |
| `PYODIDE_CDN` | CDN URL | jsdelivr |
| `SKIP_PYODIDE_BROWSER` | Skip browser tests | `0` |
| `BROWSER_TEST_TIMEOUT` | Browser test timeout | `30000` |

## Known Issues

1. **No threading**: Web Workers needed for parallelism
2. **Memory limits**: Browser memory constraints
3. **Startup time**: WASM load is slow
4. **No file I/O**: Limited filesystem access
5. **Package availability**: Not all PyPI packages work

## Related Documentation

- [Pyodide Documentation](https://pyodide.org/en/stable/)
- [Pyodide API Reference](https://pyodide.org/en/stable/usage/api/python-api.html)
- [Using Pyodide from JavaScript](https://pyodide.org/en/stable/usage/quickstart.html)

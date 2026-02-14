# prelude.py — Loaded into Pyodide at startup.
# Pre-imports and bridge functions for the Sci REPL.

import numpy as np
import json
import js  # Pyodide's JS interop module

# ---- Convenient math imports at top level ----
from math import e, inf, nan
pi = np.pi

# ---- SymPy (loaded first — its plot will be overridden by ours) ----
_SYMPY_AVAILABLE = False
try:
    from sympy import *
    _SYMPY_AVAILABLE = True
except Exception:
    pass

def _is_sympy(obj):
    if not _SYMPY_AVAILABLE:
        return False
    from sympy import Basic
    return isinstance(obj, Basic)

def _is_sympy_list(obj):
    """Check if obj is a list/tuple of SymPy expressions"""
    if not _SYMPY_AVAILABLE:
        return False
    if not isinstance(obj, (list, tuple)):
        return False
    if len(obj) == 0:
        return False
    from sympy import Basic
    return all(isinstance(item, Basic) for item in obj)

def _sympy_to_latex(obj):
    from sympy import latex as _sl
    return _sl(obj)

def _sympy_list_to_latex(obj):
    """Convert a list/tuple of SymPy expressions to LaTeX"""
    from sympy import latex as _sl
    items = [_sl(item) for item in obj]
    # Render as a LaTeX array
    return r'\begin{bmatrix}' + r' \\ '.join(items) + r'\end{bmatrix}'

# ---- Plotting bridge (defined AFTER SymPy to override its plot) ----

def plot(x, y=None, *, title="", xlabel="", ylabel="",
         label="", type="scatter", mode="lines", **kwargs):
    """Plot data using Plotly.js via the bridge.

    Usage:
        x = np.linspace(0, 2*pi, 100)
        plot(x, np.sin(x), title="Sine wave")
    """
    if y is None:
        y_data = x
        x_data = list(range(len(y_data)))
    else:
        x_data = x
        y_data = y

    if hasattr(x_data, 'tolist'):
        x_data = x_data.tolist()
    if hasattr(y_data, 'tolist'):
        y_data = y_data.tolist()

    payload = {
        "x": x_data,
        "y": y_data,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "name": label,
        "type": type,
        "mode": mode,
    }
    js.renderPlot(json.dumps(payload))


def mplot(traces, *, title="", xlabel="", ylabel="", layout=None):
    """Plot multiple traces at once."""
    processed = []
    for t in traces:
        trace = dict(t)
        for key in ("x", "y"):
            if key in trace and hasattr(trace[key], 'tolist'):
                trace[key] = trace[key].tolist()
        if "type" not in trace:
            trace["type"] = "scatter"
        if "mode" not in trace:
            trace["mode"] = "lines"
        processed.append(trace)

    payload = {
        "traces": processed,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "layout": layout or {},
    }
    js.renderPlot(json.dumps(payload))


def table(data, headers=None):
    """Render a table in the output."""
    if hasattr(data, 'tolist'):
        rows = data.tolist()
    elif isinstance(data, list) and len(data) > 0 and not isinstance(data[0], list):
        rows = [data]
    else:
        rows = data

    payload = {"rows": rows}
    if headers is not None:
        payload["headers"] = list(headers)
    js.renderTable(json.dumps(payload))


def latex(expr):
    """Render a LaTeX string in the output."""
    js.renderLatex(str(expr))

print("✓ Sci REPL ready" + (" (with SymPy)" if _SYMPY_AVAILABLE else " (NumPy only)"))

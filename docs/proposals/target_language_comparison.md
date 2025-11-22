# Target Language Comparison Matrix

**Date:** 2025-11-21
**Version:** 1.0
**Related:** orchestration_architecture.md, python_target_language.md

---

## Overview

This document provides a comprehensive comparison of UnifyWeaver's target languages to help developers choose the right target for their use case. Each target has unique strengths and fits specific deployment scenarios.

---

## Quick Reference: When to Use Each Target

| Use Case | Recommended Target | Rationale |
|----------|-------------------|-----------|
| Text processing, file manipulation | **Bash** | Native strength, ubiquitous |
| Windows enterprise integration | **C#** | .NET ecosystem, LINQ queries |
| Data science, ML, scientific computing | **Python** | Rich libraries (NumPy, Pandas, scikit-learn) |
| Complex logic, rules engines | **Prolog** | Native logic programming |
| Android/Termux development | **Bash** or **Python** | Both work well on Termux |
| Cross-platform orchestration | **Bash** (Linux/Mac) / **PowerShell** (Windows) | Shell scripts as glue |
| In-process Python from Prolog | **Python (Janus)** | Zero overhead, shared memory |
| Heavy data querying | **C# (Query Runtime)** | LINQ-optimized |

---

## Detailed Comparison

### 1. Execution Model

| Target | Mode | Process Model | Startup Cost | Runtime Overhead |
|--------|------|---------------|--------------|------------------|
| **Bash** | Subprocess | New shell process | Low (~5ms) | Low (native) |
| **C# Query** | In-process | Compile → Load assembly | High (~200ms) | Very low |
| **C# Codegen** | Subprocess | Compile → Execute | Very high (~1s) | Low (native) |
| **Python (Janus)** | In-process | Python bridge | Very low (<1ms) | Low (shared process) |
| **Python (Subprocess)** | Subprocess | New Python process | Medium (~50ms) | Medium (interpreter) |
| **Prolog** | Subprocess | New SWI-Prolog process | Medium (~30ms) | Medium (WAM) |

**Key Insights:**
- **Janus** has near-zero startup cost (best for small/medium tasks)
- **C# Query Runtime** amortizes compilation over many queries
- **Bash** fastest subprocess startup
- **C# Codegen** highest startup cost (full compilation)

---

### 2. Streaming & Data Processing

| Target | Stdin/Stdout | Streaming | JSON Support | Large Data | Null-Delimited |
|--------|--------------|-----------|--------------|------------|----------------|
| **Bash** | ✅✅ Native | ✅✅ Pipes | ✅ jq/awk | ✅✅ | ✅ |
| **C# Query** | ✅ | ✅ IEnumerable | ✅✅ System.Text.Json | ✅✅ LINQ | ✅ |
| **C# Codegen** | ✅ | ✅ Streams | ✅✅ System.Text.Json | ✅ | ✅ |
| **Python (Janus)** | ❌ | ✅ Generators | ✅✅ json module | ⚠️ Memory-bound | N/A |
| **Python (Subprocess)** | ✅✅ | ✅✅ Generators | ✅✅ json module | ✅✅ | ✅ |
| **Prolog** | ✅ | ✅ Lazy lists | ⚠️ Terms | ✅ | N/A |

**Key Insights:**
- **Bash + Python subprocess** best for streaming pipelines
- **C# Query Runtime** excels at in-memory datasets
- **Janus** limited to in-memory (no stdin/stdout)
- **Null-delimited JSON** standard for inter-process communication

---

### 3. Recursion Support

| Pattern | Bash | C# Query | C# Codegen | Python (Janus) | Python (Subprocess) | Prolog |
|---------|------|----------|------------|----------------|---------------------|--------|
| **Linear Recursion** | ✅ Memo | ✅✅ Runtime | ✅ Templates | ✅✅ `@lru_cache` | ✅✅ `@lru_cache` | ✅✅ Native |
| **Mutual Recursion** | ✅ Scripts | ✅ Runtime | ✅ Templates | ✅ Functions | ✅ Functions | ✅✅ Native |
| **Tree Recursion** | ✅ Lists | ✅ Runtime | ✅ Templates | ✅✅ Nested structs | ✅✅ Nested structs | ✅✅ Native |
| **Tail Optimization** | ❌ | ✅ Some cases | ✅ Some cases | ❌ | ❌ | ✅✅ Native |
| **Stack Depth** | Limited (bash) | High (CLR) | High (CLR) | Medium (1000) | Medium (1000) | Very high (configurable) |

**Key Insights:**
- **Prolog** best for deep recursion (native support)
- **Python** `functools.lru_cache` makes memoization trivial
- **C# Query Runtime** handles recursion efficiently
- **Bash** limited by subprocess overhead for recursion

---

### 4. Platform Support

| Target | Linux | macOS | Windows | Termux (Android) | WSL | Notes |
|--------|-------|-------|---------|------------------|-----|-------|
| **Bash** | ✅✅ Native | ✅✅ Native | ⚠️ Git Bash | ✅✅ Native | ✅✅ Native | Windows needs compatibility layer |
| **PowerShell** | ✅ PS7 | ✅ PS7 | ✅✅ Native | ❌ | ✅ | Cross-platform via PowerShell 7 |
| **C# Query** | ✅ .NET | ✅ .NET | ✅✅ Native | ❌ Hard | ✅ | Termux lacks dotnet SDK |
| **C# Codegen** | ✅ .NET | ✅ .NET | ✅✅ Native | ❌ Very hard | ✅ | Compilation on Termux impractical |
| **Python (Janus)** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ | Excellent cross-platform |
| **Python (Subprocess)** | ✅✅ | ✅✅ | ✅✅ | ✅✅ | ✅✅ | Works everywhere Python works |
| **Prolog** | ✅✅ | ✅✅ | ✅ | ✅✅ | ✅✅ | SWI-Prolog widely available |

**Key Insights:**
- **Python + Bash** best for Termux development
- **C#** difficult to test on Termux
- **PowerShell** recommended for Windows orchestration
- **Bash** recommended for Linux/macOS orchestration

---

### 5. Orchestration Capabilities

| Target | Orchestration Role | Location Preference | Communication | Coordination |
|--------|-------------------|---------------------|---------------|--------------|
| **Bash** | ✅✅ Primary | Same machine | Pipes, files | Shell built-ins |
| **PowerShell** | ✅✅ Primary (Windows) | Same machine | Pipes, files | Cmdlets |
| **C# Query** | ⚠️ Secondary | Same machine | JSON streams | Process API |
| **C# Codegen** | ⚠️ Secondary | Same machine | JSON streams | Process API |
| **Python (Janus)** | ✅✅ Same process | **Same process** | Direct calls | py_call |
| **Python (Subprocess)** | ✅ Worker | Same machine | JSON streams | subprocess |
| **Prolog** | ✅ Worker | Same machine | Term streams | process_create |

**Key Insights:**
- **Bash/PowerShell** designed for orchestration
- **Janus** unique in same-process execution
- **All targets** support JSON streaming for coordination
- **Location awareness** critical for optimization

---

### 6. Use Case Sweet Spots

#### Bash
**Best For:**
- Text processing (awk, sed, grep)
- File system operations
- System administration
- Orchestration/glue code
- Quick prototyping

**Examples:**
```bash
# Log rotation
find /var/log -name "*.log" -mtime +30 -exec gzip {} \;

# Data extraction
cat access.log | grep "ERROR" | awk '{print $1, $7}' | sort | uniq -c
```

---

#### C# (Query Runtime)
**Best For:**
- LINQ-style queries
- Complex joins and aggregations
- In-memory datasets (< 1GB)
- Windows-centric workflows
- Enterprise integration

**Examples:**
```csharp
// Multi-way join with aggregation
from user in users
join order in orders on user.Id equals order.UserId
join product in products on order.ProductId equals product.Id
group order by user.Name into g
select new { User = g.Key, TotalSpent = g.Sum(o => o.Amount) }
```

---

#### C# (Codegen)
**Best For:**
- Standalone compiled executables
- Performance-critical code
- Readable/hackable generated code
- When query runtime overhead unacceptable

**Examples:**
```csharp
// Generated recursive code (readable, maintainable)
public static int Factorial(int n) {
    if (n <= 1) return 1;
    return n * Factorial(n - 1);
}
```

---

#### Python (Janus)
**Best For:**
- Small to medium datasets (in-memory)
- Quick Python library access from Prolog
- Data science from Prolog
- Avoiding subprocess overhead
- Interactive workflows

**Examples:**
```python
# Call NumPy from Prolog via Janus
import numpy as np

def analyze(data):
    arr = np.array(data)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'median': float(np.median(arr))
    }
```

```prolog
% Prolog side
?- py_call(analyze([1,2,3,4,5]), Stats).
Stats = _{'mean': 3.0, 'std': 1.414, 'median': 3.0}.
```

---

#### Python (Subprocess)
**Best For:**
- Large-scale data processing
- ML model training
- Streaming pipelines
- When Janus unavailable
- Parallel multiprocessing

**Examples:**
```python
#!/usr/bin/env python3
# Streaming anomaly detection
import sys, json
from sklearn.ensemble import IsolationForest

model = IsolationForest()
# ... train model ...

for line in sys.stdin:
    if line.strip():
        record = json.loads(line)
        prediction = model.predict([record['features']])[0]
        if prediction == -1:  # Anomaly
            print(json.dumps(record), end='\0', flush=True)
```

---

#### Prolog
**Best For:**
- Complex logic/rules
- Constraint solving
- Graph algorithms
- Symbolic AI
- When Prolog runtime acceptable

**Examples:**
```prolog
% Transitive closure (natural in Prolog)
reachable(X, Y) :- edge(X, Y).
reachable(X, Z) :- edge(X, Y), reachable(Y, Z).

% Complex business rules
eligible_for_discount(Customer, Discount) :-
    customer_tier(Customer, Tier),
    purchase_history(Customer, Purchases),
    length(Purchases, Count),
    Count > 10,
    tier_discount(Tier, Discount).
```

---

### 7. Performance Characteristics

| Workload | Best Target | 2nd Best | Avoid |
|----------|-------------|----------|-------|
| **Text parsing (< 10MB)** | Bash | Python (subprocess) | C# (overhead) |
| **Text parsing (> 100MB)** | C# Query | Python (subprocess) | Bash (slow) |
| **Numerical computation** | Python (Janus + NumPy) | C# | Bash |
| **Graph traversal** | C# Query (LINQ) | Prolog | Bash |
| **JSON streaming** | Bash + jq | Python (subprocess) | Janus (no stdin) |
| **ML inference** | Python (Janus) | Python (subprocess) | C# (ML.NET setup) |
| **Complex queries (in-mem)** | C# Query | Python (Pandas) | Bash |
| **System administration** | Bash | PowerShell | Python |

---

### 8. Developer Experience

| Target | Learning Curve | Debugging | IDE Support | Generated Code Readability |
|--------|----------------|-----------|-------------|----------------------------|
| **Bash** | Low | ⚠️ Medium | Limited | ✅ Very readable |
| **C# Query** | Medium | ✅ Good (runtime errors clear) | ✅✅ Excellent (VS, Rider) | ⚠️ Opaque (LINQ expressions) |
| **C# Codegen** | Medium | ✅✅ Excellent (step through) | ✅✅ Excellent | ✅✅ Very readable |
| **Python (Janus)** | Low-Medium | ✅ Good (Python tracebacks) | ✅✅ Excellent (PyCharm, VS Code) | ✅✅ Readable |
| **Python (Subprocess)** | Low | ✅✅ Excellent (normal Python) | ✅✅ Excellent | ✅✅ Very readable |
| **Prolog** | High | ⚠️ Challenging (trace/debug) | ⚠️ Limited | ⚠️ Prolog knowledge required |

**Key Insights:**
- **Python subprocess** best developer experience (familiar debugging)
- **C# Codegen** most readable generated code
- **Prolog** steepest learning curve
- **Bash** simple but limited debugging tools

---

## Decision Framework

### Step 1: Identify Primary Requirement

```
┌─ Text/File Processing? ──→ Bash
├─ Windows Enterprise? ─────→ C#
├─ ML/Data Science? ────────→ Python
├─ Complex Logic? ──────────→ Prolog
└─ Orchestration? ──────────→ Bash/PowerShell
```

### Step 2: Consider Platform Constraints

```
┌─ Termux/Android? ──────────→ Python or Bash (not C#)
├─ Windows-only? ────────────→ PowerShell or C#
├─ Cross-platform? ──────────→ Python or Bash
└─ Resource-constrained? ────→ Bash (minimal overhead)
```

### Step 3: Optimize for Orchestration

```
┌─ Can use Janus? ───────────→ Python (Janus) preferred
├─ Large data (> 100MB)? ────→ Streaming (Subprocess)
├─ In-memory OK? ────────────→ Janus or C# Query Runtime
└─ Need process isolation? ──→ Subprocess
```

---

## Example Decision Process

### Scenario 1: Log Analysis on Termux

**Requirements:**
- Parse 50MB of log files
- Extract errors and warnings
- Run anomaly detection (ML)
- Generate summary report

**Platform:** Android (Termux)

**Decision:**
```
1. Termux → Rules out C#
2. ML required → Python ideal
3. 50MB data → Can use Janus or Subprocess
4. Recommendation:
   - Bash: Extract errors (grep/awk)
   - Python (Janus): ML anomaly detection
   - Bash: Format report
```

---

### Scenario 2: Enterprise Data Pipeline on Windows

**Requirements:**
- Query SQL database
- Join with CSV files
- Complex aggregations
- Windows Server deployment

**Platform:** Windows Server

**Decision:**
```
1. Windows → PowerShell orchestration
2. Complex queries → C# Query Runtime
3. Mixed sources → Data source integration
4. Recommendation:
   - PowerShell: Orchestration
   - C# Query: LINQ joins/aggregations
   - PowerShell: Result formatting
```

---

### Scenario 3: Scientific Computing Pipeline

**Requirements:**
- Read CSV with 1M rows
- Statistical analysis (NumPy/SciPy)
- Visualization (matplotlib)
- Linux cluster

**Platform:** Linux cluster

**Decision:**
```
1. 1M rows → Too large for Janus (memory)
2. NumPy/SciPy → Python ideal
3. Recommendation:
   - Bash: Orchestration
   - Python (Subprocess): All analysis
   - Streaming JSON for large data
```

---

## Future Targets (Proposed)

### JavaScript/Node.js
**Potential Use Cases:**
- Web API integration
- Serverless functions (AWS Lambda, Azure Functions)
- npm ecosystem access

**Challenges:**
- Async/promises model different from Prolog
- Callback hell vs sequential logic
- Type safety (TypeScript?)

---

### Rust
**Potential Use Cases:**
- Performance-critical code
- Systems programming
- Memory-safe low-level operations

**Challenges:**
- Ownership/borrowing incompatible with Prolog model
- Code generation complexity very high
- Steep learning curve

---

### Go
**Potential Use Cases:**
- Concurrent processing (goroutines)
- Cloud services (Docker, Kubernetes)
- Network services

**Challenges:**
- Static typing vs dynamic Prolog
- Channel model vs sequential logic

---

## Recommendations

### For Most Users
**Start with:** Bash (orchestration) + Python (subprocess) for data processing
- ✅ Works everywhere
- ✅ Easy to debug
- ✅ Large ecosystem
- ✅ Termux-friendly

### For Windows Users
**Use:** PowerShell (orchestration) + C# Query (queries)
- ✅ Native Windows integration
- ✅ LINQ powerful for queries
- ✅ Enterprise-friendly

### For Advanced Users
**Optimize with:** Location-aware orchestration
- ✅ Janus for small Python tasks (in-process)
- ✅ Subprocess for large data
- ✅ Automatic fallback

### For Researchers
**Explore:** Prolog target + Python (Janus)
- ✅ Logic programming + numerical computing
- ✅ Tight integration via Janus
- ✅ Best of both worlds

---

## Summary

**No single "best" target** - each has unique strengths:

- **Bash**: Ubiquitous, great for text/files
- **C#**: Powerful queries, Windows-native
- **Python (Janus)**: In-process, zero overhead
- **Python (Subprocess)**: Streaming, ML/data science
- **Prolog**: Complex logic, native Prolog runtime

**The power is in orchestration** - combining targets intelligently based on task, platform, and data size.

---

**Version:** 1.0
**Last Updated:** 2025-11-21
**See Also:**
- `docs/proposals/orchestration_architecture.md`
- `docs/proposals/python_target_language.md`

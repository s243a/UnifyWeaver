# Architectural Vision: UnifyWeaver as Polyglot Prolog Compiler

**Date:** 2025-10-25
**Context:** Economic Agent Playbook Development
**Scope:** Understanding the broader system we're building

## The Big Picture

UnifyWeaver is a **general-purpose Prolog compilation framework** that translates Prolog specifications into multiple target languages and formats.

The **Economic Agent playbook** is one feature/application of this framework, not the framework itself.

## Core Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    UnifyWeaver Framework                       │
│          Prolog-to-X Compilation Infrastructure                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ├─ Input: Prolog (logic, rules, specs)
                              │
                              ├─ Compilation Targets:
                              │   ├─ Bash (bash_executor.pl)
                              │   ├─ Python (python_source.pl)
                              │   ├─ CSV/JSON/HTTP (data sources)
                              │   ├─ Embedded Prolog (swipl via stdin)
                              │   ├─ Natural Language (LLM playbooks) ← NEW
                              │   └─ Future: SQL, JavaScript, etc.
                              │
                              └─ Application Domains:
                                  ├─ Data source compilation
                                  ├─ Recursive algorithm generation
                                  ├─ Code generation from specs
                                  ├─ Economic Agent planning ← CURRENT FEATURE
                                  ├─ Retrieval task orchestration
                                  └─ Any domain-specific task
```

## The Compilation Model

### Single Source, Multiple Targets

**Principle:** Write once in Prolog (declarative, high-level), compile to many targets (imperative, executable).

```prolog
% Source: Pure Prolog specification
strategy_selection(file_count(N), Strategy) :-
    N == 1,
    Strategy = single_file_precision.

strategy_selection(file_count(N), Strategy) :-
    N >= 2, N =< 5,
    Strategy = balanced_deep_dive.
```

**Target 1: Bash**
```bash
select_strategy() {
    if [ "$1" -eq 1 ]; then echo "single_file_precision"
    elif [ "$1" -ge 2 ] && [ "$1" -le 5 ]; then echo "balanced_deep_dive"
    fi
}
```

**Target 2: Python**
```python
def select_strategy(file_count):
    if file_count == 1: return "single_file_precision"
    elif 2 <= file_count <= 5: return "balanced_deep_dive"
```

**Target 3: Embedded Prolog**
```bash
PROLOG_KB='strategy_selection(file_count(1), single_file_precision). ...'
query_strategy() { echo "$PROLOG_KB" | swipl -q -g "..." }
```

**Target 4: Natural Language (LLM Playbook)**
```markdown
## Strategy Selection

Choose based on file count:
- **1 file**: Use single_file_precision (high accuracy for single target)
- **2-5 files**: Use balanced_deep_dive (efficient multi-file search)
```

## Existing Compilation Infrastructure

UnifyWeaver already has backends for several targets:

### 1. Bash Code Generation
**Module:** `src/unifyweaver/core/bash_executor.pl`
**Purpose:** Compile Prolog predicates to bash scripts
**Status:** ✅ Mature

### 2. Data Source Compilation
**Modules:**
- `src/unifyweaver/sources/csv_source.pl`
- `src/unifyweaver/sources/json_source.pl`
- `src/unifyweaver/sources/python_source.pl`
- `src/unifyweaver/sources/http_source.pl`

**Purpose:** Compile declarative data source specs to executable bash
**Status:** ✅ Mature

### 3. Template System
**Module:** `src/unifyweaver/core/template_system.pl`
**Purpose:** Variable substitution and template rendering
**Status:** ✅ Mature

### 4. Recursive Algorithm Compilation
**Modules:** `src/unifyweaver/core/advanced/*.pl`
**Purpose:** Generate optimized code for recursive patterns
**Status:** ✅ Mature

### 5. Dynamic Source Compiler
**Module:** `src/unifyweaver/core/dynamic_source_compiler.pl`
**Purpose:** Runtime compilation of source specifications
**Status:** ✅ Mature

## New Target: LLM Playbooks (Current Work)

### What We're Adding

**Module:** `src/unifyweaver/compilers/playbook_compiler.pl` (in progress)
**Purpose:** Compile Prolog specifications to natural language instructions for LLMs
**Status:** 🚧 In development

### Why This Target is Different

Most UnifyWeaver targets compile to **executable code** (bash, Python, etc.).

The LLM playbook target compiles to **instructions for AI agents**:
- Not directly executable by computers
- Interpreted by LLMs (Claude, Gemini, etc.)
- Must be human-readable and pedagogical
- Includes cost/latency metadata for economic reasoning

### The Compilation Pipeline for LLM Playbooks

```
Prolog Source (retrieval_task.pl)
         │
         ├─ [Parse] Extract predicates, rules, constraints
         │
         ├─ [Analyze] Identify strategies, tools, decision logic
         │
         ├─ [Profile] Measure tool costs/latencies (optional)
         │
         ├─ [Reason] Execute Prolog logic to pre-compute analysis
         │          (cost comparisons, feasibility checks, etc.)
         │
         ├─ [Generate] Convert to natural language
         │            - Explain strategies in prose
         │            - Format decision logic as guidance
         │            - Include cost tables, examples
         │            - Add fallback strategies
         │
         └─ [Output] Markdown playbook for LLM to read
```

## Use Case: Retrieval Task

The retrieval task demonstrates **multi-target compilation** from a single spec.

### Source Specification
```prolog
% retrieval_task.pl

% Tool definitions with metadata
tool(local_indexer, [
    cost(0),
    latency(500),
    quality(0.6),
    interface('local_indexer --query Q --files F --top-k K')
]).

tool(gemini_reranker, [
    cost(0.002),
    latency(2500),
    quality(0.85),
    interface('gemini_reranker --query Q --chunks C --top-k K')
]).

% Strategy definitions (workflows)
strategy(quick_triage, [
    step(1, local_indexer, all_files, top(25)),
    step(2, filter, score > 0.5, promising_files)
]).

strategy(balanced_deep_dive, [
    step(1, local_indexer, all_files, top(25)),
    step(2, gemini_reranker, results, top(10))
]).

% Decision logic (when to use each strategy)
select_strategy(Context, Strategy) :-
    file_count(Context, N),
    budget(Context, B),
    N == 1,
    Strategy = single_file_precision.

select_strategy(Context, Strategy) :-
    file_count(Context, N),
    N >= 2, N =< 5,
    Strategy = balanced_deep_dive.
```

### Compilation to Multiple Targets

```bash
# Target 1: Bash script for automation
unifyweaver compile retrieval_task.pl --target bash --output retriever.sh

# Target 2: Python library for agentRag integration
unifyweaver compile retrieval_task.pl --target python --output retriever.py

# Target 3: Embedded Prolog for runtime reasoning
unifyweaver compile retrieval_task.pl --target embedded_prolog --output smart_retriever.sh

# Target 4: LLM playbook for AI agents
unifyweaver compile retrieval_task.pl --target llm_playbook --output context_gatherer_playbook.md
```

### Why Multiple Targets Matter

**Different use cases need different execution models:**

| Scenario | Best Target | Why |
|----------|-------------|-----|
| CI/CD automation | Bash | Fast, portable, no dependencies |
| Integration with Python services | Python | Native library calls |
| Runtime cost optimization | Embedded Prolog | Dynamic reasoning about constraints |
| AI agent instruction | LLM Playbook | Human-readable, pedagogical |

## The Economic Agent Feature

The Economic Agent playbook is **one application domain** using the LLM playbook compilation target.

### What Makes it "Economic"

The playbook includes **resource awareness**:
- Tool costs (API charges, compute time)
- Latency estimates
- Quality tradeoffs
- Budget constraints

The Prolog source can **pre-compute economic analysis** before generating the playbook:

```prolog
% Prolog reasoning executed at compile-time
analyze_costs(FileCount, Budget, Analysis) :-
    findall(
        option(Strategy, Cost, Quality, Feasible),
        (   strategy(Strategy, _),
            strategy_cost(Strategy, FileCount, Cost),
            strategy_quality(Strategy, Quality),
            (Cost =< Budget -> Feasible = yes ; Feasible = no)
        ),
        Analysis
    ).

% This generates the cost comparison table in the playbook
```

**Output in playbook:**
```markdown
## Cost Analysis for 50 Files, $0.10 Budget

| Strategy | Cost | Quality | Feasible |
|----------|------|---------|----------|
| Quick Triage | $0.00 | 0.6 | ✓ Yes |
| Balanced Deep Dive | $0.002 | 0.85 | ✓ Yes |
| Single-File Precision | $0.002 | 0.95 | ✓ Yes |
```

The LLM reads this and makes informed decisions without having to do the math.

## Intermediate Forms and Multi-Stage Compilation

As John noted: "Intermediate forms are useful for further logical processing prior to giving the LLM final instructions."

### Compilation Stages

```
Stage 1: Prolog Source with Templates
├─ Contains {{ TEMPLATE_VARS }}
├─ Pure logic, no concrete values
└─ Example: {{ API_BUDGET | default: 0.10 }}

         ↓ [Substitute metrics]

Stage 2: Prolog with Concrete Values
├─ All variables resolved
├─ Still executable Prolog
└─ Example: available_budget(0.10).

         ↓ [Execute Prolog reasoning]

Stage 3: Analysis Results
├─ Cost comparisons
├─ Feasibility checks
├─ Recommendations
└─ Example: recommend_strategy(balanced_deep_dive, 0.002, 0.85).

         ↓ [Generate natural language]

Stage 4: Final LLM Playbook
├─ Markdown format
├─ Includes analysis as tables/text
├─ Pedagogical explanations
└─ Ready for LLM consumption
```

### Why Intermediate Forms Matter

**You can inject domain logic between stages:**

```prolog
compile_playbook(Source, Context, Output) :-
    % Stage 1→2: Substitute template variables
    substitute_variables(Source, Context, Resolved),

    % Stage 2→3: Execute cost/feasibility logic
    execute_prolog_reasoning(Resolved, Context, Analysis),

    % Stage 3→4: Generate natural language
    generate_llm_playbook(Analysis, Resolved, Output).
```

The **Stage 2→3 transition** is where Prolog logic runs:
- Cost optimization
- Constraint solving
- Feasibility analysis
- Recommendation generation

This **pre-computes insights** that would be difficult for an LLM to derive.

## Integration with Existing UnifyWeaver Patterns

The playbook compiler follows established UnifyWeaver patterns:

### 1. Source Specification Pattern
```prolog
% Like csv_source, json_source, etc.
:- source(playbook, context_gatherer, [
    prolog_spec('retrieval_task.pl'),
    target_audience(ai_agent),
    cost_analysis(enabled),
    style(detailed)
]).
```

### 2. Template System Integration
```prolog
% Uses existing template_system.pl
render_named_template('playbook/strategy_section', [
    strategy_name = 'Balanced Deep Dive',
    cost = '0.002',
    quality = '0.85'
], [], Output).
```

### 3. Compilation Driver Pattern
```prolog
% Follows compiler_driver.pl pattern
compile(playbook, Source, Options, Output) :-
    parse_source(Source, AST),
    optimize_for_target(AST, playbook, OptAST),
    generate_playbook(OptAST, Options, Output).
```

## Future Expansion

Once the LLM playbook target is mature, the framework can easily support:

### More Compilation Targets
- SQL (for database query generation)
- JavaScript (for web integration)
- YAML/JSON (for configuration files)
- GraphQL (for API schemas)
- State machines (for workflow engines)

### More Application Domains
- Database schema design
- API specification
- Workflow orchestration
- Business rule engines
- Test case generation

### All using the **same Prolog source → multi-target compilation** model.

## Summary

**UnifyWeaver Vision:**
- Prolog as universal specification language
- Multi-target compilation framework
- Semantic equivalence across targets

**Economic Agent Playbook:**
- One application domain
- Uses LLM playbook compilation target
- Demonstrates value of pre-computed economic analysis

**Current Work:**
- Build playbook_compiler.pl following existing patterns
- Support template substitution, profiling, Prolog reasoning, NL generation
- Integrate with template_system.pl and other existing infrastructure

**Next Steps:**
- Implement playbook compiler as new backend
- Test with retrieval task as reference application
- Generalize for other domains once pattern is established

---

**The key insight:** We're not building a playbook compiler. We're adding a **natural language target** to an existing polyglot compilation framework. The playbook is just the first application of this new target.

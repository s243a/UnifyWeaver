<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)
-->

# Incremental Compilation for UnifyWeaver

**Status:** PROPOSAL
**Author:** Claude (with s243a)
**Created:** 2025-12-25
**Target Version:** v1.2.0

---

## Table of Contents

1. [Philosophy & Motivation](#philosophy--motivation)
2. [Optionality & Configuration](#optionality--configuration)
3. [Goals & Non-Goals](#goals--non-goals)
4. [Technical Design](#technical-design)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)
7. [Open Questions](#open-questions)

---

## Philosophy & Motivation

### The Problem

UnifyWeaver compiles Prolog predicates to target languages (Bash, Go, Rust, C#, PowerShell, SQL). Currently, every compilation request recompiles everything from scratch, even if only one predicate changed. For large knowledge bases with hundreds of predicates, this creates:

1. **Wasted computation** - Regenerating identical code
2. **Slow iteration cycles** - Developers wait for full recompilation
3. **Resource waste** - CPU, memory, and I/O for unchanged code

### The Solution

Incremental compilation tracks what changed and only recompiles the minimum necessary. This follows the principle of **minimal work**: do only what's needed, nothing more.

### Design Philosophy

1. **Correctness over speed** - Never serve stale code. When in doubt, recompile.
2. **Transparency** - Users can inspect the cache and understand what's happening.
3. **Graceful degradation** - If caching fails, fall back to full compilation.
4. **Target-agnostic** - The caching layer sits above target-specific compilers.
5. **Opt-in by default, easy opt-out** - Users control when caching is used.

### Optionality & Configuration

Incremental compilation is **entirely optional**. Users can enable, disable, or bypass it at multiple levels:

| Level | Mechanism | Example |
|-------|-----------|---------|
| **Per-call** | Option in compile call | `compile_to_go(foo/2, [incremental(false)], Code)` |
| **Per-session** | Prolog flag | `set_prolog_flag(unifyweaver_incremental, false)` |
| **Global default** | preferences.pl | `default_option(incremental, true)` |
| **CLI flag** | Command-line | `./unifyweaver compile --no-cache pred/arity` |
| **Environment** | Env variable | `UNIFYWEAVER_CACHE=0 ./unifyweaver ...` |

**Default behavior:** Incremental compilation is **enabled by default** but can be disabled at any level above. The most specific setting wins (per-call > per-session > global > env).

**Cache management commands:**
```bash
# Clear all cached compilations
./unifyweaver cache clear

# Clear cache for specific target
./unifyweaver cache clear --target go

# Force fresh compilation (one-shot)
./unifyweaver compile --no-cache foo/2

# Show cache statistics
./unifyweaver cache status
```

**Graceful fallback:** If the cache is corrupted, missing, or unreadable, the system automatically falls back to fresh compilation without error. Users are notified via verbose logging if enabled.

---

## Goals & Non-Goals

### Goals

| Goal | Description |
|------|-------------|
| **G1** | Detect predicate source changes via content hashing |
| **G2** | Track inter-predicate dependencies using call graph |
| **G3** | Invalidate dependent predicates when dependencies change |
| **G4** | Cache compiled output indexed by predicate + hash |
| **G5** | Provide cache inspection and management commands |
| **G6** | Support all existing targets (Bash, Go, Rust, C#, PowerShell, SQL) |

### Non-Goals

| Non-Goal | Rationale |
|----------|-----------|
| **NG1** | Distributed/shared caches - Keep it simple, local-first |
| **NG2** | Partial predicate recompilation - Predicates are atomic units |
| **NG3** | Runtime hot-reload - Compile-time only |
| **NG4** | Cross-target cache sharing - Different targets produce different output |

---

## Technical Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request                              │
│              compile_to_X(pred/arity, Options)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 Incremental Compiler                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Hasher    │  │ Dependency  │  │   Cache Manager     │  │
│  │             │  │   Tracker   │  │                     │  │
│  │ hash_pred/2 │  │ call_graph  │  │ get_cached/3        │  │
│  │             │  │ scc_detect  │  │ store_cached/4      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │  Cache Hit    │           │  Cache Miss   │
    │  Return code  │           │  Compile &    │
    │               │           │  Store        │
    └───────────────┘           └───────────────┘
```

### Component Details

#### 1. Predicate Hasher (`incremental/hasher.pl`)

Computes a stable hash of a predicate's source code.

```prolog
%% hash_predicate(+Pred/Arity, -Hash)
%  Compute SHA-256 hash of predicate's clauses
hash_predicate(Pred/Arity, Hash) :-
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Head, Body), Clauses),
    term_hash(Clauses, Hash).

%% hash_predicate_with_deps(+Pred/Arity, -CombinedHash)
%  Hash includes dependencies for transitive invalidation
hash_predicate_with_deps(Pred/Arity, CombinedHash) :-
    hash_predicate(Pred/Arity, SelfHash),
    get_dependencies(Pred/Arity, Deps),
    findall(H, (member(D, Deps), hash_predicate(D, H)), DepHashes),
    term_hash([SelfHash|DepHashes], CombinedHash).
```

**Key decisions:**
- Use `term_hash/2` for Prolog terms (fast, deterministic)
- Include clause order in hash (reordering = change)
- Normalize variables before hashing for stability

#### 2. Dependency Tracker (existing: `call_graph.pl`, `scc_detection.pl`)

Already implemented. Provides:
- `build_call_graph/2` - Build dependency graph
- `get_dependencies/2` - Direct dependencies of a predicate
- `predicates_in_group/2` - SCCs for mutual recursion

**Enhancement needed:**
```prolog
%% get_transitive_dependents(+Pred/Arity, -Dependents)
%  Find all predicates that depend on this one (reverse graph traversal)
get_transitive_dependents(Pred/Arity, Dependents) :-
    build_reverse_call_graph(ReverseGraph),
    traverse_graph(ReverseGraph, Pred/Arity, Dependents).
```

#### 3. Cache Manager (`incremental/cache_manager.pl`)

Stores and retrieves compiled code.

```prolog
:- dynamic compilation_cache/5.
% compilation_cache(Pred/Arity, Target, Hash, Code, Timestamp)

%% get_cached(+Pred/Arity, +Target, +CurrentHash, -Code)
%  Retrieve cached code if hash matches
get_cached(Pred/Arity, Target, CurrentHash, Code) :-
    compilation_cache(Pred/Arity, Target, CurrentHash, Code, _).

%% store_cached(+Pred/Arity, +Target, +Hash, +Code)
%  Store compiled code in cache
store_cached(Pred/Arity, Target, Hash, Code) :-
    get_time(Timestamp),
    retractall(compilation_cache(Pred/Arity, Target, _, _, _)),
    assertz(compilation_cache(Pred/Arity, Target, Hash, Code, Timestamp)).

%% invalidate_cache(+Pred/Arity, +Target)
%  Remove cached entry (called when dependency changes)
invalidate_cache(Pred/Arity, Target) :-
    retractall(compilation_cache(Pred/Arity, Target, _, _, _)).

%% invalidate_dependents(+Pred/Arity, +Target)
%  Invalidate all predicates that depend on this one
invalidate_dependents(Pred/Arity, Target) :-
    get_transitive_dependents(Pred/Arity, Dependents),
    forall(member(Dep, Dependents), invalidate_cache(Dep, Target)).
```

**Cache storage options:**
1. **In-memory (dynamic predicates)** - Fast, lost on restart
2. **File-based (`.cache/` directory)** - Persistent across sessions
3. **Hybrid** - Memory with periodic flush to disk

Recommended: **Hybrid** with configurable persistence.

#### 4. Incremental Compiler Wrapper (`incremental/incremental_compiler.pl`)

Main entry point that wraps existing compilers.

```prolog
%% compile_incremental(+Pred/Arity, +Target, +Options, -Code)
%  Compile with incremental caching
compile_incremental(Pred/Arity, Target, Options, Code) :-
    % Check if incremental is disabled
    (   member(incremental(false), Options)
    ->  compile_fresh(Pred/Arity, Target, Options, Code)
    ;   % Compute current hash
        hash_predicate_with_deps(Pred/Arity, CurrentHash),
        % Try cache
        (   get_cached(Pred/Arity, Target, CurrentHash, Code)
        ->  format('[Incremental] Cache hit: ~w~n', [Pred/Arity])
        ;   % Cache miss - compile and store
            format('[Incremental] Cache miss: ~w~n', [Pred/Arity]),
            compile_fresh(Pred/Arity, Target, Options, Code),
            store_cached(Pred/Arity, Target, CurrentHash, Code),
            % Invalidate dependents (they may need recompilation)
            invalidate_dependents(Pred/Arity, Target)
        )
    ).

%% compile_fresh(+Pred/Arity, +Target, +Options, -Code)
%  Dispatch to target-specific compiler
compile_fresh(Pred/Arity, Target, Options, Code) :-
    (   Target = bash -> stream_compiler:compile_predicate(Pred/Arity, Options, Code)
    ;   Target = go -> go_target:compile_predicate_to_go(Pred/Arity, Options, Code)
    ;   Target = rust -> rust_target:compile_predicate_to_rust(Pred/Arity, Options, Code)
    ;   Target = csharp -> csharp_stream_target:compile_predicate_to_csharp(Pred/Arity, Options, Code)
    ;   Target = powershell -> powershell_compiler:compile_to_powershell(Pred/Arity, Options, Code)
    ;   Target = sql -> sql_target:compile_predicate_to_sql(Pred/Arity, Options, Code)
    ;   throw(error(unknown_target(Target), _))
    ).
```

### Cache File Format

For persistent caching, use a simple directory structure:

```
.unifyweaver_cache/
├── manifest.json           # Cache metadata
├── bash/
│   ├── foo_2_a1b2c3d4.pl   # Compiled code (hash in filename)
│   └── bar_3_e5f6g7h8.pl
├── go/
│   ├── foo_2_a1b2c3d4.go
│   └── bar_3_e5f6g7h8.go
└── ...
```

**manifest.json:**
```json
{
  "version": "1.0",
  "created": "2025-12-25T10:00:00Z",
  "entries": {
    "bash": {
      "foo/2": {
        "hash": "a1b2c3d4...",
        "file": "foo_2_a1b2c3d4.pl",
        "timestamp": "2025-12-25T10:00:00Z",
        "dependencies": ["bar/3", "baz/1"]
      }
    }
  }
}
```

### Invalidation Strategy

When predicate `P` changes:

1. Compute new hash for `P`
2. If hash differs from cached:
   a. Recompile `P`
   b. Find all predicates that call `P` (dependents)
   c. Invalidate cache entries for all dependents
   d. Dependents will be recompiled on next access

**Visualization:**
```
    foo/2 (changed)
       │
       ├──► bar/3 (calls foo) → INVALIDATE
       │       │
       │       └──► qux/1 (calls bar) → INVALIDATE
       │
       └──► baz/2 (calls foo) → INVALIDATE
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Foundation)

**Deliverables:**
- [ ] `incremental/hasher.pl` - Predicate hashing
- [ ] `incremental/cache_manager.pl` - In-memory cache
- [ ] Basic tests for hash stability

**Estimated effort:** 1-2 sessions

### Phase 2: Dependency Integration

**Deliverables:**
- [ ] Enhance `call_graph.pl` with reverse graph traversal
- [ ] `get_transitive_dependents/2` implementation
- [ ] Invalidation cascade logic

**Estimated effort:** 1 session

### Phase 3: Compiler Wrapper

**Deliverables:**
- [ ] `incremental/incremental_compiler.pl` - Main wrapper
- [ ] Integration with one target (Bash) as proof of concept
- [ ] `incremental(true/false)` option support

**Estimated effort:** 1-2 sessions

### Phase 4: Multi-Target Support

**Deliverables:**
- [ ] Extend to all targets (Go, Rust, C#, PowerShell, SQL)
- [ ] Target-specific cache namespacing
- [ ] Cross-target independence verification

**Estimated effort:** 1 session

### Phase 5: Persistence

**Deliverables:**
- [ ] File-based cache storage
- [ ] `manifest.json` management
- [ ] Cache load/save on startup/shutdown
- [ ] Cache directory configuration

**Estimated effort:** 1-2 sessions

### Phase 6: CLI & Management

**Deliverables:**
- [ ] `./unifyweaver cache status` - Show cache stats
- [ ] `./unifyweaver cache clear [target]` - Clear cache
- [ ] `./unifyweaver cache inspect <pred>` - Show cache entry
- [ ] Verbose mode showing cache hits/misses

**Estimated effort:** 1 session

### Phase 7: Documentation & Polish

**Deliverables:**
- [ ] User documentation
- [ ] Update FUTURE_WORK.md
- [ ] Performance benchmarks
- [ ] Edge case handling

**Estimated effort:** 1 session

---

## Testing Strategy

### Unit Tests

```prolog
test_hash_stability :-
    % Same predicate should produce same hash
    hash_predicate(foo/2, H1),
    hash_predicate(foo/2, H2),
    H1 == H2.

test_hash_change_detection :-
    % Adding a clause should change hash
    hash_predicate(foo/2, H1),
    assertz(foo(new, clause)),
    hash_predicate(foo/2, H2),
    H1 \== H2,
    retract(foo(new, clause)).

test_cache_hit :-
    store_cached(foo/2, bash, hash123, "echo foo"),
    get_cached(foo/2, bash, hash123, Code),
    Code == "echo foo".

test_cache_miss_wrong_hash :-
    store_cached(foo/2, bash, hash123, "echo foo"),
    \+ get_cached(foo/2, bash, hash456, _).

test_invalidation_cascade :-
    % bar calls foo, qux calls bar
    % Invalidating foo should invalidate bar and qux
    store_cached(foo/2, bash, h1, "foo"),
    store_cached(bar/3, bash, h2, "bar"),
    store_cached(qux/1, bash, h3, "qux"),
    invalidate_dependents(foo/2, bash),
    \+ get_cached(bar/3, bash, h2, _),
    \+ get_cached(qux/1, bash, h3, _).
```

### Integration Tests

1. **Full recompilation correctness** - Incremental output matches full compilation
2. **Cross-target isolation** - Changing Bash cache doesn't affect Go
3. **Persistence round-trip** - Save cache, restart, load cache, verify hits

### Benchmark Tests

```prolog
benchmark_incremental :-
    % Compile 100 predicates, change 1, measure recompilation time
    setup_100_predicates,
    time(compile_all_fresh),      % Baseline
    change_one_predicate,
    time(compile_all_incremental). % Should be much faster
```

---

## Open Questions

### Q1: Hash Algorithm Choice

**Options:**
- `term_hash/2` (SWI-Prolog built-in) - Fast, may have collisions
- SHA-256 via foreign interface - Slower, cryptographically secure
- MD5 - Middle ground

**Recommendation:** Start with `term_hash/2`, upgrade if collisions observed.

### Q2: When to Persist Cache?

**Options:**
- On every store (safest, slowest)
- On explicit save command
- On graceful shutdown
- Periodic background flush

**Recommendation:** Periodic flush (every 60 seconds) + shutdown save.

### Q3: Cache Size Limits?

**Options:**
- Unlimited (rely on disk space)
- LRU eviction at N entries
- Time-based expiration

**Recommendation:** Start unlimited, add LRU if needed.

### Q4: Handling Options Changes?

If compilation options change (e.g., `unique(true)` vs `unique(false)`), should we:
- Include options in hash (safest)
- Ignore options (might serve wrong code)
- Maintain separate caches per option set

**Recommendation:** Include options in hash.

---

## Appendix: Existing Code to Leverage

### call_graph.pl (existing)
```prolog
build_call_graph/2      % Build dependency graph
get_dependencies/2      % Direct dependencies
is_self_recursive/1     % Self-recursion check
predicates_in_group/2   % SCC detection
```

### scc_detection.pl (existing)
```prolog
find_sccs/2             % Tarjan's algorithm
```

### template_system.pl (existing caching pattern)
```prolog
cache_template/2        % Store template
get_cached_template/2   % Retrieve template
clear_template_cache/0  % Clear all
```

---

## References

- [Make](https://www.gnu.org/software/make/) - Classic dependency-based build
- [Bazel](https://bazel.build/) - Content-addressable caching
- [ccache](https://ccache.dev/) - Compiler cache for C/C++
- [Turborepo](https://turbo.build/) - Monorepo build caching

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-25 | Initial proposal |

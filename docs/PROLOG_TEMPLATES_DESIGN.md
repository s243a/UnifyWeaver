# Prolog Templates: Design Philosophy and Path Forward

**Status:** 🤔 Design Exploration
**Created:** 2025-11-16
**Purpose:** Thinking through when and how to use template-based generation for Prolog target

## The Core Question

When compiling Prolog predicates to Prolog (as a fallback or primary target), **when should we use templates vs. verbatim copying?**

Unlike Bash (where we transpile Prolog → Shell scripts), the Prolog target copies Prolog → Prolog. This raises philosophical questions about:
- What does compilation mean when source = target language?
- When do constraints require code transformation?
- How do we preserve vs. optimize semantics?

## Case Study: The `unique` Constraint

The `unique(true)` constraint specifies that results should be deduplicated. Let's explore how to handle this in Prolog.

### Current Behavior (Verbatim Copying)

```prolog
% User writes:
:- compile_recursive(ancestor/2, [unique(true)], Code).

% Current Prolog target generates:
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

**Result:** The constraint is **ignored**. The generated Prolog behaves exactly like the source.

**Question:** Is this acceptable?

### Option 1: Ignore Constraints (Current Approach)

**Philosophy:** "Prolog target is a verbatim fallback"

**Implementation:**
- Copy predicates as-is using `clause/2` + `portray_clause/1`
- Ignore all constraints (unique, unordered, etc.)
- User gets exact source behavior

**Pros:**
- ✅ Simple and reliable
- ✅ Preserves original semantics exactly
- ✅ No risk of introducing bugs
- ✅ Works for any valid Prolog code
- ✅ Fast compilation (no transformation)

**Cons:**
- ❌ Constraints have no effect
- ❌ User expectations may not match reality
- ❌ Inconsistent with Bash target (which enforces constraints)

**When to use:**
- Prolog as emergency fallback
- Quick prototyping
- When constraints are advisory, not mandatory

### Option 2: Template-Based Wrapper Generation

**Philosophy:** "Prolog target should honor constraints via code transformation"

**Implementation:**
Generate wrapper that enforces uniqueness:

```prolog
% Generated code:
ancestor_impl(X, Y) :- parent(X, Y).
ancestor_impl(X, Y) :- parent(X, Z), ancestor_impl(Z, Y).

ancestor(X, Y) :-
    % Collect all solutions and deduplicate
    setof([X, Y], ancestor_impl(X, Y), Solutions),
    member([X, Y], Solutions).
```

**Alternative (streaming with check):**
```prolog
:- dynamic ancestor_seen/2.

ancestor(X, Y) :-
    ancestor_impl(X, Y),
    \+ ancestor_seen(X, Y),  % Check if already seen
    assertz(ancestor_seen(X, Y)).  % Mark as seen
```

**Pros:**
- ✅ Enforces uniqueness constraint
- ✅ Consistent with Bash target behavior
- ✅ Meets user expectations

**Cons:**
- ❌ Changes execution model (collect-all vs. backtracking)
- ❌ Potential performance issues (collect all solutions)
- ❌ More complex code generation
- ❌ May break for infinite predicates
- ❌ State management issues (dynamic predicates)

**When to use:**
- When constraints are mandatory
- Finite solution spaces
- User explicitly requests constraint enforcement

### Option 3: Dialect-Specific Optimizations

**Philosophy:** "Use native Prolog features when available"

**SWI-Prolog Implementation:**
```prolog
:- table ancestor/2.  % Automatic memoization + deduplication

ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
```

**GNU Prolog Implementation:**
```prolog
% GNU Prolog doesn't have tabling, fall back to explicit dedup
% (Use Option 2's wrapper approach)
```

**Pros:**
- ✅ Uses native Prolog features (efficient)
- ✅ Idiomatic for each dialect
- ✅ No manual wrapper code
- ✅ Leverages compiler optimizations

**Cons:**
- ❌ Dialect-specific (not portable)
- ❌ Tabling changes semantics (memoization)
- ❌ Not all dialects support needed features
- ❌ More complex code generation logic

**When to use:**
- Targeting specific Prolog dialect
- Performance is critical
- User understands dialect-specific behavior

### Option 4: Configurable Strategy

**Philosophy:** "Let users choose their trade-offs"

**Configuration:**
```prolog
:- assertz(prolog_target:constraint_handling(unique, ignore)).     % Current
:- assertz(prolog_target:constraint_handling(unique, wrapper)).    % Option 2
:- assertz(prolog_target:constraint_handling(unique, native)).     % Option 3
:- assertz(prolog_target:constraint_handling(unique, warn)).       % Warn user
```

**Pros:**
- ✅ Flexible for different use cases
- ✅ User makes the trade-off decision
- ✅ Can start conservative (ignore) and evolve
- ✅ Allows experimentation

**Cons:**
- ❌ More configuration complexity
- ❌ User must understand implications
- ❌ Need to implement multiple strategies

**When to use:**
- Supporting diverse use cases
- Gradual feature rollout
- Power users who need control

## Broader Constraint Philosophy

### Constraints in Different Targets

| Constraint | Bash | C# | Prolog (Current) | Prolog (Proposed) |
|-----------|------|----|--------------------|-------------------|
| `unique(true)` | `sort -u` or hash dedup | `Distinct()` | Ignored | Configurable |
| `unique(false)` | No dedup | No `Distinct()` | Natural | Natural |
| `unordered(true)` | Can use `sort` | Can use `OrderBy()` | Natural | Natural |
| `unordered(false)` | Preserve order | Preserve order | Natural | Natural |

**Key insight:** Bash/C# need explicit code because they don't have backtracking. Prolog has backtracking naturally.

**Question:** Should Prolog target enforce constraints, or is backtracking enough?

### Semantic Preservation vs. Optimization

**Two competing philosophies:**

#### Philosophy A: "Prolog Target = Semantic Preservation"
- Goal: Generate code that behaves **exactly** like source
- Constraints are metadata for other targets
- Prolog target ignores constraints (uses natural backtracking)
- Prolog is the "reference implementation" fallback

#### Philosophy B: "Prolog Target = Constraint Enforcement"
- Goal: Generate code that **honors all constraints**
- Constraints are part of the specification
- Prolog target transforms code to enforce them
- Consistency across all compilation targets

**Which philosophy aligns with UnifyWeaver's goals?**

## When Templates are Needed

Based on this analysis, templates for Prolog target are needed when:

### 1. **Enforcing Constraints**
- User expects `unique(true)` to deduplicate in Prolog
- Requires wrapper generation or tabling directives
- **Template needed:** Yes

### 2. **Dialect-Specific Optimizations**
- Using SWI-Prolog's tabling for transitive closures
- Using GNU Prolog's FD constraints
- **Template needed:** Yes (dialect-specific)

### 3. **Runtime Library Integration**
- Predicates using partitioning need setup code
- Predicates using data sources need imports
- **Template needed:** Maybe (already handled by imports)

### 4. **Performance Wrappers**
- Adding memoization for expensive predicates
- Adding profiling/timing instrumentation
- **Template needed:** Yes (but optional)

### 5. **Simple Predicates**
- Just copy the source code
- No constraints, no special features
- **Template needed:** No (verbatim copying sufficient)

## Proposed Path Forward

### Phase 1: Document Current Behavior (Immediate)
```prolog
% Add to prolog_target.pl documentation:
%
% CONSTRAINT HANDLING:
% The Prolog target currently uses VERBATIM COPYING, which means:
% - Constraints (unique, unordered) are IGNORED
% - Generated code behaves exactly like source
% - This is intentional for fallback/reference implementation
%
% To enforce constraints, use Bash or C# targets.
% Future: Configurable constraint handling (see PROLOG_TEMPLATES_DESIGN.md)
```

### Phase 2: Add Configuration System (Short-term)
```prolog
:- dynamic prolog_target_mode/1.

% Modes:
% - fallback: Verbatim copying, ignore constraints (default)
% - enforce: Use templates to enforce constraints
% - native: Use dialect-specific features when available
prolog_target_mode(fallback).

set_prolog_target_mode(Mode) :-
    retractall(prolog_target_mode(_)),
    assertz(prolog_target_mode(Mode)).
```

### Phase 3: Implement Template Infrastructure (Medium-term)
```prolog
% Detect when templates are needed
needs_template(Pred/Arity, unique_wrapper) :-
    prolog_target_mode(enforce),
    predicate_constraints(Pred/Arity, Constraints),
    member(unique(true), Constraints).

% Generate from template vs. copy
generate_predicate_code(Pred/Arity, Options, Code) :-
    (   needs_template(Pred/Arity, TemplateType)
    ->  generate_from_template(TemplateType, Pred/Arity, Options, Code)
    ;   copy_predicate_clauses(Pred/Arity, Code)  % Verbatim fallback
    ).
```

### Phase 4: Implement Specific Templates (Long-term)
```prolog
% Template for unique constraint
template('prolog/unique_wrapper', [
    '{{predicate}}_impl{{args}} :- {{original_body}}.',
    '',
    '{{predicate}}{{args}} :-',
    '    setof({{args_list}}, {{predicate}}_impl{{args}}, Solutions),',
    '    member({{args_list}}, Solutions).'
]).

% Template for SWI-Prolog tabling
template('prolog/swi_tabling', [
    ':- table {{predicate}}/{{arity}}.',
    '',
    '{{predicate}}{{args}} :- {{original_body}}.'
]).
```

## Recommendations

### For v0.1 (Current Release)
- ✅ Keep verbatim copying as default
- ✅ Document constraint behavior clearly
- ✅ Focus on dialect system (already complete)
- ❌ Don't implement templates yet

### For v0.2 (Next Release)
- Add `prolog_target_mode` configuration
- Implement `unique` constraint template (Option 2)
- Add warnings when constraints are ignored
- Test with real use cases

### For v0.3+ (Future)
- Dialect-specific optimizations (tabling, FD)
- Performance wrappers (memoization, profiling)
- Integration with advanced features

## Open Questions

1. **Should Prolog target enforce constraints by default?**
   - Pro: Consistency across targets
   - Con: Changes semantics, may break code

2. **How to handle infinite predicates?**
   - `setof` approach fails for infinite solutions
   - Need streaming deduplication?

3. **What about other constraints?**
   - `unordered(false)`: Order matters in Prolog?
   - `optimization(speed)`: Use tabling? Compile?

4. **Should we warn users?**
   - When compiling to Prolog with constraints?
   - When constraints are ignored?

5. **Integration with preferences system?**
   - `prefer([prolog])` with `optimization(speed)` → Use GNU compilation?
   - `prefer([prolog])` with `unique(true)` → Generate wrapper?

## Conclusion

**Current State:** Prolog target uses verbatim copying (no templates)

**Philosophy:** Prolog target is a **semantic preservation fallback**, not an optimizing compiler

**Path Forward:**
1. Document current behavior clearly
2. Add configuration for future modes
3. Implement templates when specific use cases emerge
4. Start with `unique` constraint as first template
5. Evolve based on real-world needs

**Key Principle:** "Optimize for reliability first, performance second"

The Prolog target's strength is that it **always works** (verbatim copy of valid Prolog). Templates add power but also complexity and potential bugs. We should add them incrementally, driven by concrete use cases.

## See Also

- [Prolog Dialect System](PROLOG_DIALECTS.md)
- [Prolog as Fallback](PROLOG_AS_FALLBACK.md)
- [Template System Implementation](../src/unifyweaver/core/template_system.pl)
- [Constraint System Documentation](../docs/constraints/)

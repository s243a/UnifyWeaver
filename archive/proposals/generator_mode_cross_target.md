# Proposal: Extend Generator‑Mode Capabilities to All UnifyWeaver Targets

## 1. Goal
Provide **feature parity** between the Python generator target and all other UnifyWeaver code‑generation targets (C# Query Target, Bash Target, and future language targets). The new capabilities will include:

| Feature | Python (already) | C# Query Target | Bash Target |
|---------|-------------------|-----------------|------------|
| **Relation tracking** – `FrozenDict`‑style payload with a `relation` field | ✅ | ✅ | ✅ (via JSON objects) |
| **Stratified negation** – `\+ Goal` / `not(Goal)` support | ✅ | ✅ | ✅ (via set‑difference) |
| **Robust variable mapping** – `build_variable_map/2`‑style mapping of Prolog variables to target‑specific accessors | ✅ | ✅ | ✅ (via Bash associative arrays) |
| **Integration test** – compile the same Datalog program to both targets and compare outputs | ✅ (manual) | ✅ (to be added) | ✅ (to be added) |

---

## 2. High‑Level Architecture
```
Prolog source
   │
   ├─ compile_predicate_to_python/…   → Python generator code (existing)
   ├─ compile_predicate_to_csharp/…   → C# generator code (new)
   └─ compile_predicate_to_bash/…     → Bash generator script (new)
```
All three compilation pipelines will share **common helper predicates** defined in `src/unifyweaver/targets/common_generator.pl` (new file). This module will expose:

* `build_variable_map(+GoalSources, -VarMap)` – language‑agnostic mapping of Prolog variables → placeholder tokens (`{fact.arg0}`, `{join1.arg2}`, …).  
* `translate_relation_check(+Relation, +VarMap, -CheckExpr)` – generates a language‑specific predicate that checks the relation field.  
* `translate_negation(+Goal, +VarMap, -NegExpr)` – produces the “not‑in‑total” test in the target language.

Each target‑specific module will import `common_generator.pl` and then **instantiate** the placeholders into concrete syntax (Python, C#, Bash).

---

## 3. Implementation Plan per Target

### 3.1 C# Query Target (`src/unifyweaver/targets/csharp_query_target.pl`)
| Step | Description | Code Changes |
|------|-------------|--------------|
| **3.1.1** | Add `common_generator.pl` import and call `build_variable_map/2` from the existing rule‑translation predicates (`translate_copy_rule_with_builtins`, `translate_binary_join_with_constraints`, `translate_nway_join`). | Extend existing `translate_*` clauses to accept `VarMap` and pass it to `translate_builtin`. |
| **3.1.2** | **Relation tracking** – Extend `FrozenDict` equivalent (`Dictionary<string, object>`) to include a `"relation"` entry. Update `GenerateFactRule`, `GenerateCopyRule`, and join rule generators to emit `relation` in the constructed dictionary. | Modify `generate_fact_rule/4`, `generate_copy_rule/5`, etc. |
| **3.1.3** | **Negation** – Implement `translate_negation/3` that emits C# code similar to: <br>`if (!total.Contains(new FrozenDict { ["relation"] = "pred", … }))` | Add a new clause to `translate_builtin/3` for `\+ Goal` and `not(Goal)`. |
| **3.1.4** | **Variable mapping** – Use the generic `VarMap` to replace placeholders with `fact["arg0"]`, `join1["arg2"]`, etc. | In the string templates for generated C# code, replace `{ACCESS}` with the appropriate dictionary lookup. |
| **3.1.5** | **Integration test** – Add a new test file `tests/core/test_csharp_generator.pl` that compiles a small Datalog program (e.g., transitive closure with negation) to both Python and C# and asserts that the JSON output streams are identical. | Use existing `run_test/1` harness; compare the two generated files line‑by‑line. |
| **3.1.6** | **Documentation** – Update `docs/guides/csharp_query_target.md` with a “Generator Mode” section, showing the `relation` field and negation example. | Add a new subsection. |

**Challenges & Mitigations**
* C# already has a `FrozenDict` class; we only need to add the `relation` property.  
* Ensure the `HashSet<FrozenDict>` correctly implements `Equals`/`GetHashCode` for the new field.

---

### 3.2 Bash Target (`src/unifyweaver/targets/bash_target.pl`)
| Step | Description | Code Changes |
|------|-------------|--------------|
| **3.2.1** | Add `common_generator.pl` import. Bash code generation currently emits a **procedural** loop; we will replace it with a **semi‑naïve fixpoint** loop similar to Python. | Create a new `generate_fixpoint_loop/3` that writes a Bash `while` loop. |
| **3.2.2** | **Relation tracking** – Represent each fact as a JSON object (one per line) and store them in a temporary file. Use `jq` or Bash associative arrays to read the `relation` field. | In rule functions, output `{"relation":"pred","arg0":"...","arg1":"..."}`. |
| **3.2.3** | **Negation** – Implement “not‑in‑total” by checking that a JSON object does **not** appear in the current total file. This can be done with `grep -F` on the serialized JSON line or with `jq` set subtraction. | Example Bash snippet: <br>`if ! grep -F "$negated_json" "$total_file" >/dev/null; then …` |
| **3.2.4** | **Variable mapping** – Use Bash associative arrays (`declare -A fact`) where keys are `arg0`, `arg1`, … and `relation`. The generic `VarMap` will be instantiated as `${fact[arg0]}`, `${join1[arg2]}`, etc. | Replace placeholder tokens in generated Bash scripts accordingly. |
| **3.2.5** | **Fixpoint loop** – Maintain two temporary files: `total.txt` (all facts) and `delta.txt` (new facts). Each iteration reads `delta.txt`, applies all rule functions, writes new facts to `new_delta.txt`, then merges. | Use `mv new_delta.txt delta.txt` and `cat delta.txt >> total.txt`. |
| **3.2.6** | **Integration test** – Extend `tests/core/test_bash_generator.pl` (new file) that runs the generated Bash script via `bash -c` and captures its stdout. Compare against the Python version’s output. | Use `run_command` to execute the script and `assert_equal` on the JSON streams. |
| **3.2.7** | **Documentation** – Add a “Generator Mode (Bash)” section to `docs/guides/bash_target.md`, describing the JSON line format, the fixpoint loop, and the negation semantics. | New subsection with example. |

**Challenges & Mitigations**
* Bash lacks native sets; we emulate them with files and `grep`. Performance will be acceptable for modest data sizes (the target is mainly for quick prototyping).  
* JSON handling: rely on `jq` (already a dependency for other Bash utilities) for parsing and comparison. If `jq` is not available, fall back to simple string matching.

---

## 4. Shared Components (`common_generator.pl`)
| Predicate | Purpose | Language‑specific instantiation |
|-----------|---------|---------------------------------|
| `build_variable_map(+GoalSources, -VarMap)` | Walk the list of goals (fact, join_1, …) and produce a map `Var → AccessToken`. | For Python: `fact.get('arg0')`; for C#: `fact["arg0"]`; for Bash: `${fact[arg0]}`. |
| `translate_relation_check(+Relation, +VarMap, -CheckExpr)` | Generates a boolean expression that asserts `relation == <Relation>` in the target language. | Python: `fact.get('relation') == 'parent'`; C#: `fact["relation"] == "parent"`; Bash: `[ "${fact[relation]}" = "parent" ]`. |
| `translate_negation(+Goal, +VarMap, -NegExpr)` | Constructs the “not‑in‑total” test using the target’s set representation. | Python: `FrozenDict.from_dict({...}) not in total`; C#: `!total.Contains(new FrozenDict{...})`; Bash: `! grep -F "$negated_json" "$total_file"`. |
| `instantiate_placeholders(+Template, +VarMap, -Result)` | Replaces `{ACCESS}` tokens in a code template with the proper accessor string. | Simple `replace/3` loop. |

These predicates will be **unit‑tested** in `tests/core/test_common_generator.pl` to guarantee that the same logical mapping works across languages.

---

## 5. Integration Test Design
1. **Datalog program** (stored in `examples/generator_cross_test.pl`):
```prolog
% Facts
parent(john, mary).
parent(mary, sue).
parent(sue, alice).

% Rules
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Negation example
not_ancestor(X, Y) :- person(X), \+ ancestor(X, Y).
```
2. **Test steps** (in `tests/core/test_cross_generator.pl`):
   * Compile to Python (`compile_predicate_to_python/3`) → `ancestor_py.py`.  
   * Compile to C# (`compile_predicate_to_csharp/3`) → `Ancestor.cs`.  
   * Compile to Bash (`compile_predicate_to_bash/3`) → `ancestor.sh`.  
   * Run each generated artifact on the same JSONL input (`examples/parents.jsonl`).  
   * Capture stdout (JSONL stream) from each run.  
   * Sort and diff the three outputs; they must be identical.
3. **Success criteria** – No differences in the final fact sets (including `relation` fields) across all three targets.

---

## 6. Timeline (≈ 4 weeks)
| Week | Milestone |
|------|-----------|
| **1** | Create `common_generator.pl`; add unit tests for variable map and relation/negation helpers. |
| **2** | Implement C# generator extensions (relation, negation, variable mapping). Add C# integration test. |
| **3** | Implement Bash generator extensions (JSON line format, fixpoint loop, relation, negation). Add Bash integration test. |
| **4** | Write cross‑target integration test, run full suite, fix any regressions, update documentation, perform code review. |
| **5** *(optional buffer)* | Performance benchmarking for Bash (large inputs) and C# (hash‑set optimizations). |

---

## 7. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| **Bash performance** – file‑based set emulation may be slow on large datasets. | Medium | Document size limits; optionally fall back to Python for heavy workloads. |
| **C# `FrozenDict` equality** – adding `relation` may break existing hash semantics. | Low | Update `Equals`/`GetHashCode` to include `relation`. Add regression tests. |
| **Cross‑target test flakiness** – differing JSON ordering. | Low | Sort output before diff; canonicalize JSON (e.g., `jq -S`). |
| **Missing `jq` on user machines** (Bash target). | Medium | Provide a pure‑Bash fallback using string matching; warn in docs. |

---

## 8. Deliverables
* **Source code** – Updated `csharp_query_target.pl`, `bash_target.pl`, new `common_generator.pl`.  
* **Tests** – Unit tests for common helpers, target‑specific tests, and a cross‑target integration test.  
* **Documentation** – Updated guides for C# and Bash generator mode, plus a high‑level design note (`PROPOSALS/generator_mode_cross_target.md`).  
* **Release notes** – Mention “Generator‑mode parity across targets” and list new capabilities.

---

### Closing
By centralizing the variable‑mapping and relation/negation logic in a language‑agnostic module, we keep the implementation DRY, simplify future extensions (e.g., adding a Java target), and guarantee that all targets behave identically. The proposed timeline is realistic for a small team and provides clear checkpoints for review.

**Next step:** Approve the proposal (or suggest modifications) so we can start work on `common_generator.pl` and the C# extensions.

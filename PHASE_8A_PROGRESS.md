# Phase 8a: Database Query/Filter Predicates - Work in Progress

## Status: Implementation Complete, Debugging in Progress

This document tracks the implementation of Phase 8a: Adding database filtering support using native Prolog constraint syntax.

## Completed Work

### 1. Core Infrastructure (✅ Complete)

#### Constraint Extraction (`go_target.pl:1695-1730`)
- `extract_db_constraints/3` - Recursively extracts filter constraints from predicate body
- `is_comparison_constraint/1` - Identifies supported comparison operators (>, <, >=, =<, =, \=)
- Separates `json_record/1` from comparison constraints
- Handles implicit AND (multiple constraints in conjunction)

#### Filter Code Generation (`go_target.pl:1731-1782`)
- `generate_filter_checks/3` - Generates Go if statements for all constraints
- `constraint_to_go_check/3` - Converts each Prolog constraint to Go code
- Maps Prolog operators to Go equivalents:
  - `>` → `>`
  - `<` → `<`
  - `>=` → `>=`
  - `=<` → `<=`
  - `=` → `==`
  - `\=` → `!=`

#### Term-to-Go Expression Conversion (`go_target.pl:1784-1811`)
- `field_term_to_go_expr/3` - Converts Prolog terms to Go expressions
- Handles variables (maps to `fieldN` using identity check)
- Handles literals (atoms, numbers)
- Returns "unknownVar" for unmapped variables with warning

#### Database Read Mode Integration (`go_target.pl:1879-2031`)
- Modified `compile_database_read_mode/4` to support filtered reads
- `generate_field_extractions_for_read/2` - Extracts fields as `interface{}`
- `generate_output_for_read/3` - Generates JSON output with selected fields only
- Supports both filtered reads (when body has constraints) and unfiltered reads

### 2. Test Suite (✅ Created)

#### Test File: `test_db_filters.pl`
- **Test 0**: Populate database (write mode) - ✅ Working
- **Test 1**: Age filter (`Age >= 30`)
- **Test 2**: Multi-field filter (`Age > 25, City = "NYC"`)
- **Test 3**: Salary range (`30000 =< Salary =< 80000`)
- **Test 4**: Not equal (`City \= "NYC"`)
- **Test 5**: All comparison operators
- **Test 6**: No filter (baseline)

#### Test Runner: `run_filter_tests.sh`
- Generates all Go programs from Prolog
- Creates test data (10 sample users)
- Builds and runs each test
- Verifies output

### 3. Bug Fixes (✅ Applied)

1. **Discontiguous Predicates**: Added `:- discontiguous term_to_go_expr/3.`
2. **Format Error**: Changed `term_to_go_expr` to always return atoms (not raw numbers)
3. **Duplicate Functions**: Removed duplicate `generate_field_extractions_for_read` and `generate_output_for_read`
4. **Variable Mapping Bug**: Simplified field mapping extraction (removed faulty `pairs_keys_values` call)
5. **Name Conflict**: Renamed to `field_term_to_go_expr/3` to avoid conflict with existing `term_to_go_expr/3`

## Current Issue

### Problem: Silent Compilation Failure

**Symptoms**:
- Test 0 (populate/write mode) generates successfully ✅
- Tests 1-6 (read mode with filters) fail silently after extracting constraints ❌
- No error message is generated
- Predicate appears to fail (return false) rather than throw exception

**Debug Output**:
```
=== Test 1: Age Filter (Age >= 30) ===
=== Compiling adults/2 to Go ===
  Mode: Database read (bbolt)
  Predicate body: json_record([name-_9958,age-_9960,city-_10010,salary-_10022]),_9960>=30
  Field mappings: [name-_9958,age-_9960,city-_10010,salary-_10022]
  Constraints: [_9960>=30]
[Compilation stops here - no output generated]
```

**Likely Cause**:
One of the following predicates is failing:
1. `generate_field_extractions_for_read/2`
2. `generate_filter_checks/3`
3. `generate_output_for_read/3`
4. Database read code generation in `compile_database_read_mode/4`

**Testing Results**:
- ✅ `generate_field_extractions_for_read/2` works in isolation
- ❓ `generate_filter_checks/3` produces code but variable mapping may be incorrect
- ❓ `generate_output_for_read/3` not tested in isolation yet
- ❓ Full pipeline integration failing

**Variable Mapping Concern**:
The identity check `Term == Var` in `field_term_to_go_expr/3` may not be finding variables correctly due to how `clause/2` creates fresh variable instances.

## Next Steps

### Immediate (Debug Current Issue)

1. **Add Explicit Debugging**:
   ```prolog
   format('DEBUG: About to generate field extractions~n'),
   generate_field_extractions_for_read(FieldMappings, ExtractCode),
   format('DEBUG: Generated extractions: ~w chars~n', [ExtractCode]),
   ```

2. **Test Each Helper in Isolation**:
   - Test `generate_output_for_read/3` with sample data
   - Verify variable identity checks work with `clause/2` variables
   - Check if format strings are valid

3. **Alternative Approach**:
   - If variable identity fails, use position-based matching instead
   - Create a mapping from variable positions to field names
   - Match constraint variables by position rather than identity

### After Fix

4. **Complete Test Suite**: Verify all 6 filter tests generate and run correctly
5. **Key Optimization Detection**: Auto-detect when constraints can use key-based lookup
6. **Documentation**: Update `GO_JSON_FEATURES.md` with Phase 8a examples
7. **PR Creation**: Commit and create pull request

## Design Decisions

### Why Native Prolog Syntax?
Using Prolog's built-in comparison operators (>, <, >=, =<, =, \=) provides:
- **SQL Compatibility**: Direct mapping to WHERE clauses for future SQL target
- **No New DSL**: Developers already know the syntax
- **Type Awareness**: Works with existing schema validation
- **Composability**: Easy to add OR, NOT later

### Why Separate Read/Write Modes?
- Write mode: JSON input → validate → store in database
- Read mode: Database → filter → JSON output (selected fields only)
- Clean separation of concerns
- Different code generation patterns

### Field Selection
Predicates only output fields mentioned in the head:
```prolog
adults(Name, Age) :-  % Only outputs name and age
    json_record([name-Name, age-Age, city-City, salary-Salary]),
    Age >= 30.
```

Output:
```json
{"name": "Alice", "age": 35}
```

## Files Modified

| File | Lines Added | Lines Removed | Status |
|------|-------------|---------------|--------|
| `src/unifyweaver/targets/go_target.pl` | ~300 | ~5 | ✅ Modified |
| `test_db_filters.pl` | 131 | 0 | ✅ Created |
| `run_filter_tests.sh` | 125 | 0 | ✅ Created |
| `PHASE_8A_PROGRESS.md` | This file | 0 | ✅ Created |

## References

- **User Requirements**: Start simple (Phase 8a), auto-detect optimizations, SQL-ready design
- **Phase 8b** (Future): String operations (`=@=`, `contains`), list membership (`member/2`)
- **Phase 8c** (Future): Fail-on-error mode, rollback transactions
- **SQL Target** (Future): Generate `WHERE` clauses from same constraint syntax

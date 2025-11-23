# feat(python): Add multi-clause support and streaming verification

## Summary
This PR enhances the Python target with multi-clause (disjunction/OR-choice) support and adds end-to-end streaming verification tests.

## Changes

### Multi-Clause Compilation
- **Refactored** `compile_predicate_to_python/3` to handle multiple clauses per predicate
- **Design**: Each Prolog clause compiles to a separate Python generator function (`_clause_N`)
- **Benefits**: Proper variable scoping, clean control flow, supports OR-choice semantics

### Code Generation Improvements
- Fixed atom translation to emit JSON strings (`"c1"` instead of Python identifier `c1`)
- Enhanced `get_dict/3` translation to handle constant values with early return
- Changed control flow from `continue` to `return` (correct for generator functions)

### End-to-End Testing
- **New test suite**: `tests/core/test_python_execution.pl`
  - `streaming_behavior`: Verifies incremental JSONL processing (unbuffered I/O)
  - `multi_clause_execution`: Verifies OR-choice produces multiple outputs
- **Updated**: `tests/core/test_python_target.pl` with `compile_multiple_clauses` test

## Example

**Prolog**:
```prolog
parent(R, c1) :- get_dict(id, R, 1).
parent(R, c2) :- get_dict(id, R, 1).
```

**Generated Python**:
```python
def _clause_0(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('id') != 1: return
    yield "c1"

def _clause_1(v_0: Dict) -> Iterator[Dict]:
    if v_0.get('id') != 1: return
    yield "c2"

def process_stream(records: Iterator[Dict]) -> Iterator[Dict]:
    for record in records:
        yield from _clause_0(record)
        yield from _clause_1(record)
```

**Input**: `{"id": 1}`  
**Output**: `"c1"\n"c2"\n`

## Testing
✅ All 7 tests passing:
- 5 code generation tests (`test_python_target.pl`)
- 2 execution tests (`test_python_execution.pl`)

## Files Modified
- `src/unifyweaver/targets/python_target.pl` - Core compiler enhancements
- `tests/core/test_python_target.pl` - Added multi-clause test
- `tests/core/test_python_execution.pl` - New execution test suite

## Related
- Builds on #68 (initial Python target implementation)
- Part of Python target Phase 2: Multi-Clause & Streaming

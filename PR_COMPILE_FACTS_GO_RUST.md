# feat: Add compile_facts/3 public API to Go and Rust targets

## Summary

This PR adds standalone `compile_facts/3` public APIs to the Go and Rust targets, completing the compile_facts feature across all 12 UnifyWeaver targets.

## Changes

### Go Target (`go_target.pl`)
- **New API:** `compile_facts_to_go(+Pred, +Arity, -GoCode)`
- Generates struct-based fact exports with:
  - `GetAllPRED() []PRED` - Returns all facts as slice
  - `StreamPRED(fn func(PRED))` - Iterator with callback
  - `ContainsPRED(target PRED) bool` - Membership check

### Rust Target (`rust_target.pl`)
- **New API:** `compile_facts_to_rust(+Pred, +Arity, -RustCode)`
- Generates `#[derive(Debug, Clone, PartialEq, Eq)]` struct exports with:
  - `get_all_pred() -> Vec<PRED>` - Returns all facts as Vec
  - `stream_pred() -> impl Iterator<Item = PRED>` - Iterator
  - `contains_pred(target: &PRED) -> bool` - Membership check

### Documentation Updates
- `docs/GO_TARGET.md` - Added API reference with examples
- `docs/RUST_TARGET.md` - Added API reference with examples
- `education/book-06-go-target/01_introduction.md` - Feature table + Quick Start
- `education/book-09-rust-target/01_introduction.md` - Feature table + Quick Start

## Usage Examples

### Go
```prolog
?- ['examples/family_tree'].
?- go_target:compile_facts_to_go(parent, 2, Code).
```

### Rust
```prolog
?- ['examples/family_tree'].
?- rust_target:compile_facts_to_rust(parent, 2, Code).
```

## Testing

- ✅ Both implementations generate valid code
- ✅ Existing Go target tests pass
- ✅ No regressions in other targets

## compile_facts Feature Matrix (12 targets)

| Target | Status |
|--------|--------|
| Java | ✅ |
| Kotlin | ✅ |
| Scala | ✅ |
| Clojure | ✅ |
| Jython | ✅ |
| C | ✅ |
| C++ | ✅ |
| Python | ✅ |
| SQL | ✅ |
| PowerShell | ✅ |
| **Go** | ✅ **NEW** |
| **Rust** | ✅ **NEW** |

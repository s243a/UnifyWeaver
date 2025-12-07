# PR Title

```
Add CROSS JOIN support to SQL Target (Phase 3e)
```

# PR Description

## Summary

Fixes a bug in `generate_from_clause_multi` and adds explicit **CROSS JOIN** syntax for Cartesian products (tables with no shared variables).

This completes the SQL Target JOIN feature set - all standard SQL JOIN types are now supported.

## Changes

### Bug Fix
- Fixed `findall/3` syntax error in `generate_from_clause_multi` (line ~1219)
- Was: `findall(TN, member(G, List), (functor(...)), Result)` (broken)
- Now: `findall(TN, (member(G, List), functor(...)), Result)` (correct)

### New Feature
- Added `generate_cross_joins/2` helper predicate
- Generates explicit `CROSS JOIN` syntax instead of implicit comma-separated tables

## Example

**Prolog:**
```prolog
color_size_combos(Color, Size) :-
    colors(_, Color),
    sizes(_, Size).
```

**Generated SQL:**
```sql
SELECT colors.name, sizes.size
FROM colors
CROSS JOIN sizes;
```

## Test Results

```
✅ Test 1: Simple CROSS JOIN (2 tables)
✅ Test 2: Triple CROSS JOIN (3 tables)
✅ Test 3: Mixed INNER + CROSS JOIN
✅ All previous JOIN tests pass (backwards compatible)
```

## Files Changed

- `src/unifyweaver/targets/sql_target.pl` - Core fix (+11/-3 lines)
- `test_sql_cross_join.pl` - Test cases (new, 76 lines)
- `proposals/CROSS_JOIN_DESIGN.md` - Design doc (new)

## JOIN Support Complete

| JOIN Type | Status | Pattern |
|-----------|--------|---------|
| INNER JOIN | ✅ | Shared variables between tables |
| LEFT OUTER JOIN | ✅ | `A, (B ; null)` |
| RIGHT OUTER JOIN | ✅ | `(A ; null), B` |
| FULL OUTER JOIN | ✅ | `(A ; null), (B ; null)` |
| CROSS JOIN | ✅ | No shared variables (this PR) |

## Breaking Changes

None. Pure bug fix enabling previously broken feature.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

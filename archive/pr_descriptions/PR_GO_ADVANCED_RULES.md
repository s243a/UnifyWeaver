# PR Title

Add advanced rule compilation to Go target: match+body predicates and multi-rule support

# PR Description

This PR adds two major enhancements to the Go target compilation that significantly expand its capabilities:

## 1. Match + Body Predicate Support

Enables rules that combine regex pattern matching with source predicates:

```prolog
parsed(Name, Level, Message) :-
    log_entry(Name, Line),
    match(Line, '([A-Z]+): (.+)', auto, [Level, Message]).
```

**Implementation:**
- Capture groups from match constraints can now be mapped to head arguments
- Field references and capture references coexist in output expressions
- Position-based mapping determines which variables come from fields vs captures

## 2. Multiple Rules with Different Body Predicates

Compiles predicates with multiple rules having different source arities:

```prolog
person(Name) :- user(Name).
person(Name) :- employee(Name, _).
person(Name) :- contractor(Name, _, _).
```

**Implementation:**
- Sequential rule matching using if-continue pattern based on field count
- Selective field assignment (only assigns fields actually used in output)
- Supports match constraints and arithmetic constraints in rule bodies
- Smart import detection for strings package when needed

## Key Technical Features

- **Selective field assignment**: Analyzes output expressions to determine which fields are used, avoiding Go's "declared and not used" errors
- **Smart imports**: Detects when strings/regexp/strconv packages are needed based on code patterns
- **Code reuse**: Both features share capture mapping and constraint handling logic
- **Comprehensive testing**: Includes test files demonstrating all new capabilities

## Documentation

- Added Go Target section to main README.md
- Updated docs/GO_TARGET.md with:
  - Two new examples (Examples 9 & 10)
  - Updated feature lists and limitations
  - Expanded Supported Patterns quick reference
  - Moved completed features from "Planned" to "Current"

## Test Files
- `test_match_body.pl` - Match + body predicate combinations
- `test_multi_body.pl` - Multiple rules with different bodies

## Files Modified
- `src/unifyweaver/targets/go_target.pl` - Core implementation (+169 lines)
- `README.md` - Added Go Target section
- `docs/GO_TARGET.md` - Comprehensive documentation updates

These enhancements make the Go target compilation much more flexible for real-world data transformation tasks.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

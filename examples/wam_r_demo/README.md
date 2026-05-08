# wam_r_demo -- genealogy

End-to-end demo of the WAM R hybrid target. Five family-tree facts,
a recursive ancestor relation, a foreign predicate (`age_of/2`)
implemented in R, and a `findall/3` aggregator. Together they
exercise every major mechanism of the target in one ~80-line Prolog
file:

- WAM-compiled facts (`parent_of/2`)
- Recursive Prolog (`ancestor/2`)
- Mixed WAM + foreign body (`ancestor_age_above/2`)
- Foreign handler returning bindings (`age_of/2`)
- Compiled `findall/3` (`all_ancestors/2`)

## Build

From the repo root:

```bash
swipl -g build_demo -t halt examples/wam_r_demo/genealogy.pl
```

That writes a self-contained R project to `./_genealogy_out_r/`.
No compile step is needed -- `Rscript` loads the generated source
directly.

## Run

The generated program's `main` takes `<pred>/<arity>` followed by
the args. CLI args are parsed via the runtime's operator-precedence
parser, so atoms (`alice`), integers (`60`), lists (`[a, b, c]`),
structures (`f(a, b)`), and arithmetic expressions (`3+2*2`) all
work directly:

```bash
cd _genealogy_out_r

# Direct facts
Rscript R/generated_program.R 'parent_of/2' alice bob
# -> true

# Recursive ancestor
Rscript R/generated_program.R 'ancestor/2' alice carol
# -> true (alice -> bob -> carol)

Rscript R/generated_program.R 'ancestor/2' alice frank
# -> true (alice -> eve -> frank)

Rscript R/generated_program.R 'ancestor/2' bob alice
# -> false (other direction)

# Foreign handler -- returns the age bound to its second arg
Rscript R/generated_program.R 'age_of/2' bob 60
# -> true

Rscript R/generated_program.R 'age_of/2' bob 30
# -> false (handler returned 60, doesn't unify with 30)

# Mixed WAM + foreign body
Rscript R/generated_program.R 'ancestor_age_above/2' alice 50
# -> true (alice has an ancestor whose age > 50; bob is 60)

Rscript R/generated_program.R 'ancestor_age_above/2' carol 50
# -> false (carol's only descendant in the tree, dan, is 12)

# findall over the recursive ancestor -- list arg parses via the CLI
# parser
Rscript R/generated_program.R 'all_ancestors/2' alice '[bob,carol,dan,eve,frank]'
# -> true
```

## What's where

- [genealogy.pl](genealogy.pl) -- the entire demo: facts, predicates,
  R foreign handler, and the `build_demo/0` that calls
  `write_wam_r_project/3`.
- `_genealogy_out_r/` -- generated after `build_demo` runs
  (gitignored).

The handler source is built from a single Prolog string in
`age_handler_code/1` -- a compact way to inject R code without
external `.R` files. For larger handlers, declare a separate
`r_foreign_handlers([handler(P/A, "...")])` entry per predicate.

## See also

- [docs/WAM_R_TARGET.md](../../docs/WAM_R_TARGET.md) -- full
  options and builtin reference.
- [tests/test_wam_r_generator.pl](../../tests/test_wam_r_generator.pl)
  -- runs every documented feature end-to-end.

# wam_scala_demo — genealogy

End-to-end demo of the WAM Scala hybrid target. Five family-tree
facts, a recursive ancestor relation, and a foreign predicate
(`age_of/2`) implemented entirely on the Scala side. Together they
show every major mechanism of the target in one ~50-line Prolog file:

- WAM-compiled facts (`parent_of/2`)
- Recursive Prolog (`ancestor/2`)
- Mixed WAM + foreign body (`ancestor_age_above/2`)
- Foreign handler returning bindings (`age_of/2`)

## Build

From the repo root:

```bash
swipl -g build_demo -t halt examples/wam_scala_demo/genealogy.pl
```

That writes a self-contained SBT project to `./_genealogy_out/`.
Compile it with `scalac` (no SBT required for the demo):

```bash
cd _genealogy_out
mkdir classes
scalac -d classes src/main/scala/demo/genealogy/*.scala
```

## Run

The generated `main` takes `<pred>/<arity>` followed by the args:

```bash
# Direct facts
scala -classpath classes demo.genealogy.GeneratedProgram \
  'parent_of/2' alice bob
# → true

# Recursive ancestor
scala -classpath classes demo.genealogy.GeneratedProgram \
  'ancestor/2' alice carol
# → true (alice → bob → carol)

scala -classpath classes demo.genealogy.GeneratedProgram \
  'ancestor/2' alice dan
# → true (alice → bob → carol → dan)

scala -classpath classes demo.genealogy.GeneratedProgram \
  'ancestor/2' bob alice
# → false (other direction)

# Foreign handler — returns the age bound to its second arg
scala -classpath classes demo.genealogy.GeneratedProgram \
  'age_of/2' bob 60
# → true

scala -classpath classes demo.genealogy.GeneratedProgram \
  'age_of/2' bob 30
# → false (handler returned 60, doesn't unify with 30)

# Mixed WAM + foreign body
scala -classpath classes demo.genealogy.GeneratedProgram \
  'ancestor_age_above/2' alice 50
# → true (some ancestor of alice has age > 50; bob does)

scala -classpath classes demo.genealogy.GeneratedProgram \
  'ancestor_age_above/2' carol 50
# → false (carol's only ancestor in the tree is dan, age 12)
```

## What's where

- [genealogy.pl](genealogy.pl) — the entire demo: facts, predicates,
  foreign handler, and the `build_demo/0` that calls
  `write_wam_scala_project/3`.
- `_genealogy_out/` — generated after `build_demo` runs (gitignored).

The handler source is built from a single Prolog string in
`age_handler_code/1` — a compact way to inject Scala code without
external `.scala` files. For larger handlers, a separate
`scala_foreign_handlers([handler(P/A, "...")])` entry per predicate
keeps things readable.

## See also

- [docs/WAM_SCALA_TARGET.md](../../docs/WAM_SCALA_TARGET.md) — full
  options and builtin reference.
- [tests/test_wam_scala_runtime_smoke.pl](../../tests/test_wam_scala_runtime_smoke.pl)
  — runs every documented feature end-to-end.

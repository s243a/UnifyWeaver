# CSV Data Source Playbook

## Purpose
Validate that UnifyWeaver can ingest a CSV file as a declarative source, expose it as a predicate, and compile downstream logic into runnable Bash scripts. This playbook doubles as a regression test for dynamic sources (e.g., CSV plugins) and the compiler driver’s ability to mix dynamic and static predicates.

## Inputs & Artifacts
- Source definition: `csv_pipeline.pl`
- Example documentation: `playbooks/csv_data_source_playbook.md`
- Test data: `test_data/test_users.csv`
- Output directory (configurable): `output/csv_playbook`

## Prerequisites
1. SWI-Prolog installed (`swipl` available).
2. Working directory: repository root or the sandbox checkout.
3. The CSV plugin auto-registers via `src/unifyweaver/sources/csv_source.pl` once `csv_pipeline.pl` is loaded.

## Execution Steps
1. Load the pipeline and compiler driver, then compile the target predicate:
   ```bash
   /mnt/c/Program\ Files/swipl/bin/swipl.exe \
     -q \
     -g "['csv_pipeline.pl','src/unifyweaver/core/compiler_driver.pl'], \
         compiler_driver:compile(get_user_age/2, [output_dir('output/csv_playbook')], Scripts), \
         format('Generated: ~w~n',[Scripts]), halt" \
     -t halt
   ```
2. Verify the command reports both scripts (dynamic source + derived predicate):
   ```text
   Generated: [output/csv_playbook/users.sh,output/csv_playbook/get_user_age.sh]
   ```
3. Inspect the generated source wrapper if needed:
   ```bash
   cat output/csv_playbook/users.sh
   ```
4. Run the compiled predicate to stream user data (defaults to `sort -u`):
   ```bash
   bash output/csv_playbook/get_user_age.sh
   ```

## Verification
- `users.sh` should contain an AWK-based CSV reader that preserves header-aware parsing.
- `get_user_age.sh` should invoke `users_stream | sort -u` and emit `id:name:age` rows.
- Running the predicate with no stdin should print the deduplicated dataset from `test_users.csv`.

## Troubleshooting
| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Unknown procedure: compiler_driver:compile/3` | `compiler_driver.pl` not loaded in the SWI goal | Include it in the `-g` load list (see step 1). |
| `test_users.csv: No such file` | Command run from wrong directory | Execute from repo root or pass an absolute path in the `csv_file/1` option. |
| Empty output | `sort -u` deduped everything or CSV data missing | Inspect `test_data/test_users.csv`; ensure duplicates exist if you expect multiples. |

## Related Material
- Human-readable playbook narrative: `playbooks/csv_data_source_playbook.md`
- Principles and workflow guidance: `education/book-workflow/examples_library/data_source_playbooks.md`

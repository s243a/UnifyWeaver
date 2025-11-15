# Playbook Record Parsers

Playbooks reference example records instead of embedding scripts. Use one of the parsers below (in preference order) to extract a record into a temporary file before execution.

| Priority | Tool | Path / Install | Notes |
| --- | --- | --- | --- |
| 1 | Perl Extractor | `scripts/utils/extract_records.pl` | Default choice. Supports `--format`, `--query`, `--file-filter`. Documented in [skills/skill_extract_records.md](../../skills/skill_extract_records.md). |
| 2 | Python Extractor | `docs/playbooks/parsing/extract_records.py` (future) | Placeholder for a dependency-free Python implementation. Until it lands, fall back to option 3. |
| 3 | `parsc` | [GitHub: parsc](https://github.com/s243a/parsc) | External CLI from @s243a. Use when Perl isn’t available; convert Markdown to JSON and filter records programmatically. |

## Usage Pattern (Perl Extractor)
```bash
scripts/utils/extract_records.pl \
  --format content \
  --query "record.name.here" \
  playbooks/examples_library/example_file.md \
  > tmp/extracted.sh
```

Set `TMP_FOLDER` (and related env vars) before executing the extracted script. Clean up the temporary artifacts when done.

## Adding New Parsers
1. Implement the parser (e.g., Python) in this directory.
2. Provide a short README entry describing invocation, dependencies, and compatibility.
3. Update `skills/skill_extract_records.md` to mention the new option/reference.

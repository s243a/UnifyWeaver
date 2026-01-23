# Skill: Pipe Communication

Generate TSV and JSON pipe readers and writers for inter-target Unix pipeline communication.

## When to Use

- User asks "how do I connect different languages via pipes?"
- User needs ETL pipeline components
- User wants Unix-style data streaming
- User needs to generate code that reads/writes TSV or JSON Lines

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/pipe_glue').

% Generate TSV writer (producer)
generate_pipe_writer(python, [name, age, score], [], Code).

% Generate TSV reader (consumer)
generate_pipe_reader(awk, [name, age, score], [], Code).

% Generate JSON Lines writer
generate_pipe_writer(go, [id, name, value], [format(json)], Code).
```

## Supported Targets

| Target | TSV Read | TSV Write | JSON Read | JSON Write |
|--------|----------|-----------|-----------|------------|
| `awk` | Yes | Yes | Yes | Yes |
| `python` | Yes | Yes | Yes | Yes |
| `bash` | Yes | Yes | No | No |
| `go` | Yes | Yes | Yes | Yes |
| `rust` | Yes | Yes | Yes | Yes |

## Writer Generation (Producer Side)

### TSV Writers

```prolog
generate_tsv_writer(Target, Fields, Code).
```

**AWK:**
```prolog
generate_tsv_writer(awk, [name, age], Code).
% Output: print name "\t" age
```

**Python:**
```prolog
generate_tsv_writer(python, [name, age], Code).
% Output: print("\t".join(str(x) for x in [name, age]))
```

**Bash:**
```prolog
generate_tsv_writer(bash, [name, age], Code).
% Output: echo -e "${name}\t${age}"
```

**Go:**
```prolog
generate_tsv_writer(go, [name, age], Code).
% Output: fmt.Printf("%s\t%s\n", name, age)
```

**Rust:**
```prolog
generate_tsv_writer(rust, [name, age], Code).
% Output: println!("{}\t{}", name, age);
```

### JSON Writers

```prolog
generate_json_writer(Target, Fields, Code).
```

**Python:**
```prolog
generate_json_writer(python, [name, age], Code).
```

**Output:**
```python
import json
print(json.dumps({"name": name, "age": age}))
```

**Go:**
```prolog
generate_json_writer(go, [name, age], Code).
```

**Output:**
```go
jsonBytes, _ := json.Marshal(struct{Name, Age}{name, age})
fmt.Println(string(jsonBytes))
```

## Reader Generation (Consumer Side)

### TSV Readers

```prolog
generate_tsv_reader(Target, Fields, Code).
```

**AWK:**
```prolog
generate_tsv_reader(awk, [name, age, score], Code).
% Fields accessible as $1, $2, $3
```

**Python:**
```prolog
generate_tsv_reader(python, [name, age, score], Code).
```

**Output:**
```python
import sys
for line in sys.stdin:
    fields = line.rstrip("\n").split("\t")
    name, age, score = fields[0], fields[1], fields[2]
    # Process record here
```

**Bash:**
```prolog
generate_tsv_reader(bash, [name, age, score], Code).
```

**Output:**
```bash
while IFS=$'\t' read -r name age score; do
    # Process record here
done
```

**Go:**
```prolog
generate_tsv_reader(go, [name, age, score], Code).
```

**Output:**
```go
scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    fields := strings.Split(scanner.Text(), "\t")
    name, age, score := fields[0], fields[1], fields[2]
    // Process record here
}
```

### JSON Readers

```prolog
generate_json_reader(Target, Fields, Code).
```

**Python:**
```prolog
generate_json_reader(python, [id, name, value], Code).
```

**Output:**
```python
import sys, json
for line in sys.stdin:
    record = json.loads(line)
    id, name, value = record["id"], record["name"], record["value"]
    # Process record here
```

## Pipeline Orchestration

### Generate Pipeline Script

```prolog
generate_pipeline_script(Steps, Options, Script).
```

**Step Format:**
- `step(Name, local, Script)` - Local processing
- `step(Name, remote, URL)` - Remote HTTP call

**Basic Example:**
```prolog
generate_pipeline_script([
    step(extract, local, 'cat data.tsv'),
    step(transform, local, 'python transform.py'),
    step(analyze, remote, 'http://api.example.com/analyze')
], [language(bash)], Script).
```

**Options:**
- `language(Lang)` - Output language (python, bash, go)
- `error_handling(Mode)` - `fail_fast` (default), `continue`, `retry(N)`
- `timeout(Seconds)` - Per-step timeout

### Step Execution Order

Steps execute sequentially. Output from each step pipes to the next:

```
step1 → stdout | stdin → step2 → stdout | stdin → step3
```

### Error Handling

```prolog
generate_pipeline_script([
    step(fetch, remote, 'http://api.example.com/data'),
    step(process, local, 'python process.py'),
    step(store, local, 'python store.py')
], [language(bash), error_handling(fail_fast)], Script).
```

| Mode | Behavior |
|------|----------|
| `fail_fast` | Stop pipeline on first error (default) |
| `continue` | Log error, continue with next step |
| `retry(N)` | Retry failed step N times before failing |

### Conditional Steps

Use `when/2` to make steps conditional:

```prolog
generate_pipeline_script([
    step(fetch, local, 'curl -s $URL'),
    step(validate, local, 'python validate.py'),
    step(transform, local, 'python transform.py', when(validation_passed)),
    step(fallback, local, 'python fallback.py', when(validation_failed))
], [language(bash)], Script).
```

### Complete Pipeline Example

```prolog
generate_pipeline_script([
    step(extract, local, 'python extract.py --input data.csv'),
    step(clean, local, 'python clean.py --remove-nulls'),
    step(enrich, remote, 'http://api.example.com/enrich'),
    step(transform, local, 'python transform.py --format json'),
    step(load, local, 'python load.py --db postgres')
], [
    language(bash),
    error_handling(retry(3)),
    timeout(300)
], Script).
```

**Generated Bash:**
```bash
#!/bin/bash
set -euo pipefail

python extract.py --input data.csv \
  | python clean.py --remove-nulls \
  | curl -s -X POST -d @- http://api.example.com/enrich \
  | python transform.py --format json \
  | python load.py --db postgres
```

## Format Options

```prolog
generate_pipe_writer(Target, Fields, Options, Code).
generate_pipe_reader(Target, Fields, Options, Code).
```

**Options:**
- `format(tsv)` - Tab-separated values (default)
- `format(json)` - JSON Lines format

## Common Patterns

### ETL Pipeline

```prolog
% Extract (Python CSV reader)
generate_pipe_reader(python, [id, name, value], [], ExtractCode).

% Transform (AWK filtering)
generate_pipe_writer(awk, [id, name, value], [], TransformCode).

% Load (Go JSON writer)
generate_pipe_writer(go, [id, name, value], [format(json)], LoadCode).
```

**Usage:**
```bash
python extract.py | awk -f transform.awk | go run load.go
```

### Multi-Language Data Flow

```bash
# Generate code for each step
cat input.tsv \
  | python step1.py \    # Python TSV reader -> JSON writer
  | go run step2.go \    # Go JSON reader -> TSV writer
  | awk -f step3.awk     # AWK TSV processing
```

## Related

**Parent Skill:**
- `skill_ipc.md` - IPC sub-master

**Sibling Skills:**
- `skill_rpyc.md` - Network-based RPC
- `skill_python_bridges.md` - Cross-runtime embedding

**Code:**
- `src/unifyweaver/glue/pipe_glue.pl`

# Add JSON Input Support to Go Generator Mode

## Summary

Enables loading initial facts from stdin via JSONL format, making generator mode practical for real data pipelines.

## Usage

```prolog
compile_predicate_to_go(ancestor/2, [mode(generator), json_input(true)], Code)
```

```bash
echo '{"relation":"parent","args":{"arg0":"john","arg1":"mary"}}
{"relation":"parent","args":{"arg0":"mary","arg1":"sue"}}' | go run ancestor.go
```

## Changes

- `go_generator_header` now accepts Options, adds `bufio`/`os` imports conditionally
- `compile_go_generator_execution` inserts stdin JSONL parsing when enabled
- Facts from stdin merged with `GetInitialFacts()` before fixpoint iteration

## Generated Code

```go
// Load additional facts from stdin (JSONL format)
scanner := bufio.NewScanner(os.Stdin)
for scanner.Scan() {
    var fact Fact
    if err := json.Unmarshal(scanner.Bytes(), &fact); err == nil {
        total[fact.Key()] = fact
    }
}
```

## Verification

- All 6 tests pass
- Piping 3 parent facts produces all 6 expected ancestor facts

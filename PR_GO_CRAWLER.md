# feat(go): Compile Semantic Predicates (Crawler/Search)

## Summary
This PR integrates the Semantic Runtime into the Go compiler (`go_target.pl`), enabling the translation of `semantic_search/3` and `crawler_run/2` predicates into valid Go code that utilizes the `hugot` (embedding) and `bbolt` (storage) based runtime library.

## Changes
- **`go_target.pl`**:
    - Implemented `compile_semantic_rule_go` to generate `main()` logic for semantic operations.
    - Added `crawler_run` logic to initialize `PtCrawler` and start crawling.
    - Added `semantic_search` logic to embed query and search vector store.
    - Ensure necessary imports (`unifyweaver/targets/go_runtime/...`) are generated.
- **Tests**:
    - Updated `tests/core/test_go_semantic_compilation.pl` to verify `crawler_run` compilation.

## Usage
```prolog
run_crawler :- crawler_run(['http://example.com'], 3).
```

Compiles to:
```go
// ... imports ...
func main() {
    // ... init store/embedder ...
    craw := crawler.NewCrawler(store, emb)
    craw.Crawl([]string{"http://example.com"}, int(3))
}
```

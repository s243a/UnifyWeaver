# docs(guides): Add HTTP transport examples to cross-target glue guide

## Summary

Adds documentation for the HTTP transport in the Transport-Aware Compilation section.

## Changes

### [cross-target-glue.md](file:///home/s243a/Projects/UnifyWeaver/docs/guides/cross-target-glue.md)

Added "HTTP Transport for Remote Hosts" section with:
- Example step definitions targeting remote service URLs
- Generated Python code showing `requests.post` calls
- Tip about Go (`http.Client`) and Bash (`curl`) alternatives

## Example Added

```prolog
Steps = [
    step(ml_predict, python, 'http://ml-service:8080/predict', []),
    step(enrich, go, 'http://enricher:9000/enrich', [timeout(60)])
],
generate_pipeline_for_groups([group(http, Steps)], [language(python)], Code).
```

This completes the documentation for all three transport types: `pipe`, `direct`, and `http`.

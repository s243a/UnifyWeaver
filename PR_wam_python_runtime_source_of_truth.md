# PR Title

Use packaged Python WAM runtime as source of truth

# PR Description

## Summary

- makes `compile_wam_runtime_to_python/2` read the packaged `WamRuntime.py`
- removes the divergent generated fallback runtime path from Python WAM project generation
- documents that Python WAM now has one runtime source of truth

## Details

Generated Python WAM projects already copy `src/unifyweaver/targets/wam_python_runtime/WamRuntime.py` as `wam_runtime.py`. This PR makes `compile_wam_runtime_to_python/2` return that same static runtime source instead of assembling a second runtime from Prolog string fragments.

This keeps runtime inspection tests, generated projects, compiled facts, and TSV-backed dynamic fact loading on the same runtime surface. The TSV benchmark path is unchanged: generated `main.py` still loads TSV files and registers facts into the packaged runtime with `register_indexed_atom_fact2_pairs`.

## Verification

```sh
swipl -q -g run_tests -t halt tests/test_wam_python_target.pl
swipl -q -g run_tests -t halt tests/test_wam_python_effective_distance_smoke.pl
python3 -m py_compile src/unifyweaver/targets/wam_python_runtime/WamRuntime.py
swipl -q -g "use_module(src/unifyweaver/targets/wam_python_target), halt"
```

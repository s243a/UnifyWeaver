# feat(python): Semantic Playbook & Runtime Fixes

## Summary
This PR validates the Python Semantic Runtime with a real-world playbook (`examples/semantic_playbook.pl`) that indexes and searches an RDF export from Pearltrees. It includes critical fixes to the XML flattening logic and ONNX runtime configuration to ensure stability on constrained environments (e.g., Termux).

## Changes
- **Playbook:** Added `semantic_playbook.pl` and `run_semantic_demo.sh` to demonstrate end-to-end indexing and search.
- **Runtime Fixes:**
    - `crawler.py`: Fixed logic to preserve child elements (e.g., titles) when processing parent nodes, ensuring data is not lost before embedding.
    - `onnx_embedding.py`: Explicitly request `CPUExecutionProvider` to improve compatibility.
    - `python_target.pl`: Added atomic assignment support (`Var = Atom`) and fixed regex escaping for runtime inlining.

## Usage
```bash
./examples/run_semantic_demo.sh
```

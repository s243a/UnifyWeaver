# PR: Add Book 17 - WAM Target

## Summary
Adds a new book to the UnifyWeaver Education Series detailing the WAM (Warren Abstract Machine) target. This documentation explains the philosophy, instruction set, and compilation patterns for the new lower-level target.

## Chapters
- **README.md**: Overview of WAM's role as a "Universal Low-Level Fallback".
- **01_introduction.md**: Philosophy of WAM and its relationship to the "Native Lowering" approach.
- **02_isa.md**: Symbolic Instruction Set Architecture (ISA) reference for argument/temporary registers and core instructions.
- **03_compilation.md**: Practical examples of mapping Prolog facts, rules, and recursive structures to WAM.
- **04_fallback_hub.md**: Strategic overview of how WAM bridges the gap to WASM (via WAT) and JVM bytecode.

## Related
- Implementation PR in `UnifyWeaver` repo: "feat: add WAM target compiler and symbolic runtime emulator"

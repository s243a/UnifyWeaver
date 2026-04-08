# Semantic Search Interface Philosophy

## Core Principles

### 1. Declarative Independence
The logic of a program (the "What") should be decoupled from the search implementation (the "How"). A Prolog predicate like `find_similar(Query, Results)` should describe a logical relationship that remains valid whether it's executed via a Python `transformers` pipeline on a GPU or a Rust `candle` implementation on a mobile CPU.

### 2. Hardware-Aware Compilation
Hardware selection (GPU vs CPU) is a deployment concern, not a logic concern. The transpiler should be responsible for emitting the correct initialization code based on the requested `device` option, while providing a seamless fallback mechanism to ensure the generated code is resilient across different environments.

### 3. Provider Pluralism
The ML ecosystem is fragmented across languages (Python/Transformers, Rust/Candle, Go/Hugot, .NET/ONNX). UnifyWeaver embraces this by providing a unified Prolog interface that maps to the most idiomatic and performant provider for each target language, while maintaining semantic parity.

### 4. Zero-Cost Abstraction
The generic interface should not introduce runtime overhead. The mapping happens at compile-time, resulting in native, direct calls to the target's semantic libraries.

## The "Write Once, Search Everywhere" Vision
By using a declarative `semantic_provider` directive, a developer can specify a tiered strategy for their search predicates. They can prioritize high-performance GPU backends for production servers and lightweight CPU backends for edge/CLI deployments, all while keeping the core logic identical.

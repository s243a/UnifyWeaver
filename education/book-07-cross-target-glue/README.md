<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 7: Cross-Target Glue

**Multi-Language Pipelines and Cloud Deployment**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers UnifyWeaver's cross-target glue system, which enables predicates compiled to different languages to communicate and compose seamlessly. You'll learn how to build hybrid systems where each component runs in its optimal environment.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)
- At least one target book ([Book 2](../book-02-bash-target/README.md), [3](../book-03-csharp-target/README.md), [5](../book-05-python-target/README.md), or [6](../book-06-go-target/README.md))

**Recommended:**
- Multiple target books - the more targets you know, the more you can compose

**Technical:**
- SWI-Prolog 8.0+ installed
- Familiarity with at least one target language (Python, Go, Rust, or C#)

## What You'll Learn

By completing this book, you will be able to:

- Design multi-language data pipelines
- Generate glue code for shell scripts (AWK, Python, Bash)
- Create in-process .NET bridges (C#, PowerShell, IronPython)
- Orchestrate native binaries (Go, Rust)
- Build distributed systems with HTTP and socket communication
- Choose optimal communication strategies for different scenarios

## Learning Path

### Part 1: Foundations (Chapters 1-2)

**1. Introduction to Cross-Target Communication** (`01_introduction.md`)
- Why cross-target glue?
- The Unix philosophy extended
- Overview of location and transport models
- Quick start example

**2. Philosophy and Design Principles** (`02_philosophy.md`)
- Location transparency
- Sensible defaults with override
- Runtime family affinity
- Data format negotiation
- Streaming by default

### Part 2: Core Infrastructure (Chapters 3-4)

**3. Target Registry and Mapping** (`03_target_registry.md`)
- Target metadata management
- Runtime families
- Predicate-to-target declarations
- Location and transport resolution
- Default behaviors

**4. Pipe Protocols and Data Formats** (`04_pipe_protocols.md`)
- TSV protocol specification
- JSON Lines protocol
- In-process protocols
- Header negotiation
- Field mapping

### Part 3: Shell Integration (Chapters 5-6)

**5. Shell Script Generation** (`05_shell_glue.md`)
- AWK script generation
- Python script generation
- Bash script generation
- Format options (TSV, CSV, JSON)
- Header handling

**6. Shell Pipeline Orchestration** (`06_shell_pipelines.md`)
- Multi-stage pipelines
- Step configuration
- Input/output handling
- Error propagation
- Real-world examples

### Part 4: .NET Integration (Chapters 7-8)

**7. .NET Bridge Generation** (`07_dotnet_bridges.md`)
- Runtime detection
- PowerShell in-process hosting
- IronPython in-process hosting
- CPython fallback via pipes
- Choosing the right approach

**8. IronPython Compatibility** (`08_ironpython_compat.md`)
- Compatible modules
- Incompatible modules (numpy, pandas, etc.)
- Automatic runtime selection
- Migration strategies

### Part 5: Native Targets (Chapters 9-10)

**9. Go and Rust Code Generation** (`09_native_code_gen.md`)
- Go pipe-compatible wrappers
- Rust pipe-compatible wrappers
- TSV and JSON modes
- Parallel processing with goroutines
- Build script generation

**10. Native Binary Orchestration** (`10_native_orchestration.md`)
- Binary management
- Compilation on demand
- Cross-compilation (5 platforms)
- Mixed pipelines (shell + native)
- Performance optimization

### Part 6: Network Communication (Chapters 11-12)

**11. HTTP Services** (`11_http_services.md`)
- Service registry
- Go HTTP servers (net/http)
- Python Flask servers
- Rust Actix-web servers
- Consistent API format

**12. Distributed Pipelines** (`12_distributed_pipelines.md`)
- HTTP client generation
- Socket communication
- Network pipeline orchestration
- Local + remote step mixing
- Error handling strategies

### Part 7: Production Ready (Chapters 15-16)

**15. Production Deployment** (`15_deployment_production.md`)
- Service declarations and deployment
- Security enforcement (encryption for remote)
- Lifecycle management (start, stop, graceful shutdown)
- Error resilience (retry, fallback, circuit breaker)
- Monitoring (health checks, metrics, logging, alerting)

**16. Cloud & Enterprise** (`16_cloud_enterprise.md`)
- Container deployment (Docker, Kubernetes)
- Secrets management (Vault, AWS, Azure, GCP)
- Multi-region deployment and failover
- Cloud functions (Lambda, GCF, Azure Functions)
- API Gateway integration

### Part 8: Reference (Chapters 13-14)

**13. API Reference** (`13_api_reference.md`)
- Complete predicate documentation
- Option reference
- Error codes and handling
- Performance tuning

**14. Case Studies** (`14_case_studies.md`)
- ETL pipeline: Bash + AWK + Python + SQL
- .NET integration: C# + PowerShell + IronPython
- High-performance: Go + Rust
- Microservices: Distributed HTTP pipeline

## Chapter Summary

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 1 | Introduction | Location transparency, transport model |
| 2 | Philosophy | Design principles, use cases |
| 3 | Target Registry | Families, capabilities, defaults |
| 4 | Pipe Protocols | TSV, JSON, header negotiation |
| 5 | Shell Glue | AWK, Python, Bash generation |
| 6 | Shell Pipelines | Multi-stage orchestration |
| 7 | .NET Bridges | PowerShell, IronPython hosting |
| 8 | IronPython | Compatibility, fallback strategies |
| 9 | Native Code | Go/Rust wrappers, parallel processing |
| 10 | Native Orchestration | Compilation, cross-platform |
| 11 | HTTP Services | Server generation, API design |
| 12 | Distributed | Clients, sockets, mixed pipelines |
| 13 | API Reference | Complete documentation |
| 14 | Case Studies | Real-world examples |
| 15 | Production | Deployment, security, monitoring |
| 16 | Cloud & Enterprise | Containers, secrets, serverless |

## Implementation Status

The cross-target glue system is fully complete:

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (registry, mapping) | Complete |
| 2 | Shell Integration (AWK, Python, Bash) | Complete |
| 3 | .NET Integration (C#, PowerShell, IronPython) | Complete |
| 4 | Native Targets (Go, Rust) | Complete |
| 5 | Network Layer (HTTP, sockets) | Complete |
| 6 | Production Ready (deployment, monitoring) | Complete |
| 7a | Container Deployment (Docker, Kubernetes) | Complete* |
| 7b | Secrets Management (Vault, AWS, Azure, GCP) | Complete* |
| 7c | Multi-Region & Cloud Functions | Complete* |

*Phase 7 features are **EXPERIMENTAL** - code generation is tested, but actual deployment has not been integration-tested.

## Code Examples

Each chapter includes working code examples in the `examples/` directory:

```
examples/
  01-hello-pipeline/       # Simple AWK + Python pipeline
  02-shell-formats/        # TSV, CSV, JSON format examples
  03-dotnet-bridge/        # C# + PowerShell integration
  04-native-parallel/      # Go parallel processing
  05-distributed/          # HTTP microservice example
  06-cloud-deploy/         # Docker, K8s, serverless examples
```

## Additional Resources

- Design documentation: `docs/design/cross-target-glue/`
- API reference: `docs/design/cross-target-glue/04-api-reference.md`
- Source code: `src/unifyweaver/glue/`
- Tests: `tests/integration/glue/`

## What's Next?

After completing this book, you'll understand:

- How to compose predicates across language boundaries
- When to use pipes vs in-process vs network communication
- How to optimize for performance or simplicity
- How to build production-ready distributed systems

**Continue to Book: Workflow** to learn about:
- Complete application orchestration
- Deployment strategies
- Production monitoring
- CI/CD integration

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.

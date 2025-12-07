# Chapter 1: Introduction to the Prolog Target

## What is the Prolog Target?

The Prolog target is one of UnifyWeaver's transpilation backends that converts Prolog source code into executable Prolog scripts. While this might seem circular (transpiling Prolog to Prolog), it serves several important purposes.

### Why Transpile Prolog to Prolog?

At first glance, transpiling Prolog code to Prolog might seem unnecessary. However, the Prolog target provides several valuable capabilities:

#### 1. **Dialect Translation**

Different Prolog implementations (dialects) have varying syntax, capabilities, and execution models:

- **SWI-Prolog**: Rich module system, interpreted execution, extensive libraries
- **GNU Prolog**: Minimal module system, native compilation, portable binaries

The Prolog target can generate code optimized for a specific dialect, handling differences in:
- Initialization directives
- Module import syntax
- Constraint solver syntax
- Built-in predicates

#### 2. **Dependency Injection**

UnifyWeaver analyzes your code to detect dependencies on:
- Partitioners (for parallel data processing)
- Data sources (CSV, JSON, SQLite, etc.)
- Backends (execution engines)

The generated script automatically includes the necessary imports and initialization code.

#### 3. **Compilation Support**

The Prolog target can generate:
- **Interpreted scripts**: Ready to run with `swipl` or `gprolog`
- **Compiled binaries**: Native executables via GNU Prolog's `gplc` compiler

This allows you to:
- Develop with SWI-Prolog's rich tooling
- Deploy as standalone binaries with GNU Prolog
- Automatically fall back to interpreted mode if compilation fails

#### 4. **Security and Validation**

Integration with UnifyWeaver's firewall system allows:
- Restricting which dialects can be used
- Validating code against dialect capabilities
- Enforcing security policies on generated code

## Use Cases

### Standalone Executables

Generate self-contained Prolog scripts or binaries that can run without UnifyWeaver:

```prolog
% Input: factorial.pl
factorial(0, 1) :- !.
factorial(N, R) :- N > 0, N1 is N - 1, factorial(N1, R1), R is N * R1.

% Generate standalone script
?- generate_prolog_script([factorial/2],
                         [dialect(gnu), compile(true)],
                         Code),
   write_prolog_script(Code, 'factorial', [dialect(gnu), compile(true)]).

% Result: factorial (native binary)
$ ./factorial
```

### Cross-Dialect Compatibility

Write once in SWI-Prolog, deploy to GNU Prolog:

```prolog
% Developed and tested with SWI-Prolog
my_predicate(...) :- ...

% Generate GNU Prolog version with automatic adjustments
?- generate_prolog_script([my_predicate/2],
                         [dialect(gnu)],
                         GnuCode).
```

### Dependency Bundling

Automatically include UnifyWeaver modules your code depends on:

```prolog
% Your code uses partitioning
process_data(Data, Results) :-
    partitioner_init(fixed_size(rows(100)), [], Handle),
    partitioner_partition(Handle, Data, Partitions),
    % ... process partitions
    Results = Partitions.

% Generated script includes:
% :- use_module(unifyweaver(core/partitioner)).
% :- ensure_loaded(unifyweaver(core/partitioners/fixed_size)).
% Plus plugin registration
```

### Development Workflow

1. **Develop** with SWI-Prolog (rich debugging, REPL)
2. **Test** with interpreted mode (fast iteration)
3. **Deploy** as GNU Prolog binary (standalone, fast startup)
4. **Fallback** to interpreted if target platform lacks compiler

## Benefits

### For Users

- **No manual dependency management**: Automatically includes what you need
- **Cross-platform deployment**: One source, multiple targets
- **Performance options**: Choose interpreted (flexible) or compiled (fast)
- **Security**: Firewall policies protect against unsafe operations

### For Implementers

- **Clean separation**: Target-specific logic isolated in dialect modules
- **Extensible**: Easy to add new Prolog dialects
- **Testable**: Each component has clear responsibilities
- **Maintainable**: Changes to one dialect don't affect others

## The UnifyWeaver Pipeline

```
┌─────────────────┐
│  Source Prolog  │
│   (your code)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   UnifyWeaver   │
│    Analysis     │ ◄─── Detect dependencies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prolog Target  │
│   Generation    │ ◄─── Select dialect
└────────┬────────┘      Apply policies
         │               Generate code
         ▼
┌─────────────────┐
│ Generated Script│
│  (.pl file)     │
└────────┬────────┘
         │
         ├─── compile(false) ───► Executable script
         │
         └─── compile(true) ────► gplc ───► Native binary
```

## What's Next?

In the next chapter, we'll dive into the architecture of the Prolog target, exploring how it transforms your code through the transpilation pipeline.

---

**Key Takeaways:**
- The Prolog target enables dialect translation, dependency injection, and compilation
- It supports both interpreted scripts and native binaries
- Integration with the firewall provides security and validation
- The target simplifies deployment and cross-platform compatibility

---

## Navigation

**←** [Previous: The Prolog Target: Implementation and Usage Guide](00_index) | [📖 Book 11: Prolog Target](README) | [Next: Chapter 2: Architecture Overview →](02_architecture)

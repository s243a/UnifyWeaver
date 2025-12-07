# Chapter 3: Understanding Prolog Dialects

## What are Prolog Dialects?

Prolog is not a single language, but a family of related implementations. Each implementation (or "dialect") shares the core Prolog semantics but differs in:

- Available built-in predicates
- Module system design
- Execution model (interpreted vs compiled)
- Constraint solving capabilities
- Standard library features
- Performance characteristics

UnifyWeaver currently supports two major dialects:

1. **SWI-Prolog**: The most popular Prolog implementation
2. **GNU Prolog**: A standards-compliant implementation with native compilation

## SWI-Prolog

### Overview

**Website**: https://www.swi-prolog.org/
**License**: BSD-2-Clause
**Primary Use**: Development, research, web applications

### Key Capabilities

```prolog
dialect_capabilities(swi, Capabilities).
% Capabilities = [
%     name('SWI-Prolog'),
%     compilation(interpreted),
%     constraint_solver(clpfd),
%     module_system(full)
% ]
```

### Strengths

**1. Rich Module System**

SWI-Prolog has a sophisticated module system with:
- Explicit imports and exports
- Module inheritance
- Contextual module resolution
- Meta-predicates

```prolog
:- module(my_module, [exported_pred/2]).

:- use_module(library(lists)).
:- use_module(unifyweaver(core/partitioner)).

exported_pred(X, Y) :- ...
```

**2. Comprehensive Libraries**

- HTTP server and client
- JSON, XML, YAML parsing
- Database connectivity (ODBC, SQLite)
- Constraint solving (CLP(FD), CLP(R))
- Tabling/memoization
- Multithreading

**3. Development Tools**

- Interactive REPL with debugging
- Graphical debugger
- Profiler
- Code coverage analysis
- Unit testing framework

**4. Standards Compliance**

- ISO Prolog core
- Extensions for practical programming
- Well-documented deviations

### Limitations

**1. Interpreted Execution**

- Slower startup than compiled binaries
- Higher memory footprint
- Requires SWI-Prolog installation

**2. Deployment**

- Not a single-file executable
- Depends on SWI-Prolog runtime
- Larger distribution size

### When to Use SWI-Prolog

✅ **Good for:**
- Development and prototyping
- Web applications (HTTP server)
- Database integration
- Complex module structures
- Using advanced libraries
- Interactive debugging

❌ **Not ideal for:**
- Standalone executables
- Resource-constrained environments
- Fast startup time requirements
- Minimal dependencies

## GNU Prolog

### Overview

**Website**: http://www.gprolog.org/
**License**: GPL/LGPL
**Primary Use**: Portable executables, embedded systems

### Key Capabilities

```prolog
dialect_capabilities(gnu, Capabilities).
% Capabilities = [
%     name('GNU Prolog'),
%     compilation(compiled),
%     constraint_solver(fd),
%     module_system(basic)
% ]
```

### Strengths

**1. Native Compilation**

GNU Prolog includes `gplc`, a compiler that produces:
- Standalone native executables
- No runtime dependencies (can be statically linked)
- Fast startup time
- Small binary size

```bash
gplc --no-top-level factorial.pl -o factorial
./factorial
# Runs instantly, no interpreter needed
```

**2. Portability**

- Runs on many platforms (x86, ARM, etc.)
- Minimal dependencies
- Self-contained binaries
- Easy deployment

**3. Performance**

- Fast execution (compiled code)
- Low memory overhead
- Efficient constraint solving (FD)

**4. Standards Compliance**

- Strong ISO Prolog compliance
- Predictable behavior
- Fewer extensions (simpler)

### Limitations

**1. Basic Module System**

GNU Prolog has a minimal module system:
- No use_module/1 directive
- Limited module isolation
- Mostly relies on include/1

```prolog
% GNU Prolog approach
:- include('library.pl').
```

**2. Limited Libraries**

- Fewer built-in libraries than SWI
- No HTTP server
- Basic I/O
- FD constraints (not CLP(FD))

**3. Development Experience**

- Less sophisticated REPL
- Fewer debugging tools
- More manual dependency management

**4. Compilation Requirements**

- Requires C compiler
- Longer build times
- Platform-specific binaries

### When to Use GNU Prolog

✅ **Good for:**
- Standalone executables
- Deployment to end users
- Resource-constrained systems
- Fast startup requirements
- Minimal dependencies
- Portable binaries

❌ **Not ideal for:**
- Complex module hierarchies
- Heavy library usage
- Web applications
- Rapid prototyping

## Dialect Comparison Table

| Feature | SWI-Prolog | GNU Prolog |
|---------|------------|------------|
| **Execution** | Interpreted | Compiled |
| **Startup** | Slower (load interpreter) | Fast (native binary) |
| **Distribution** | Requires runtime | Single executable |
| **Module System** | Full (use_module) | Basic (include) |
| **Constraint Solving** | CLP(FD), CLP(R), CLP(Q) | FD |
| **Tabling** | Yes | No |
| **HTTP Server** | Yes (library(http)) | No |
| **Database** | SQLite, ODBC | Minimal |
| **Debugging** | Rich (graphical) | Basic |
| **REPL** | Full-featured | Basic |
| **Standards** | ISO + extensions | Strict ISO |
| **Threading** | Yes | Limited |
| **Binary Size** | Large (with runtime) | Small (standalone) |
| **Development Speed** | Fast (interpreted) | Slower (compile) |
| **Production Deployment** | Requires installation | Self-contained |

## Initialization Differences

One critical difference is how programs are initialized:

### SWI-Prolog

```prolog
% For scripts
:- initialization(main, main).

% For compiled applications
:- initialization(Goal).
```

**Behavior**:
- `initialization/2`: Runs Goal when loaded as script
- First argument: goal to run
- Second argument: when to run (main = on startup)

### GNU Prolog (Interpreted)

```prolog
% For interpreted mode
:- main.
```

**Behavior**: Executes goal immediately when file is consulted

### GNU Prolog (Compiled)

```prolog
% For compiled binaries (gplc --no-top-level)
:- initialization(main).
```

**Behavior**: Sets entry point for binary execution

**Critical**: This is what UnifyWeaver's fix addresses! The dialect generation must choose the right initialization based on compile mode.

## Choosing the Right Dialect

### Decision Matrix

```
┌─────────────────────────────────────────────────┐
│ Do you need standalone executables?            │
├─────────────────────────────────────────────────┤
│ YES → Use GNU Prolog (compile)                  │
│ NO  → Continue ↓                                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Do you need advanced libraries (HTTP, DB)?     │
├─────────────────────────────────────────────────┤
│ YES → Use SWI-Prolog                            │
│ NO  → Continue ↓                                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Do you need complex module structures?         │
├─────────────────────────────────────────────────┤
│ YES → Use SWI-Prolog                            │
│ NO  → Continue ↓                                │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Is startup time critical?                      │
├─────────────────────────────────────────────────┤
│ YES → Use GNU Prolog (compile)                  │
│ NO  → Either works, prefer SWI for development  │
└─────────────────────────────────────────────────┘
```

### Common Patterns

**Pattern 1: Develop with SWI, Deploy with GNU**

```prolog
% Development
swipl my_app.pl

% Generate GNU Prolog version for deployment
?- generate_prolog_script([my_app/0],
                         [dialect(gnu), compile(true)],
                         Code),
   write_prolog_script(Code, 'my_app_gnu.pl',
                      [dialect(gnu), compile(true)]).

% Result: my_app_gnu (standalone binary)
```

**Pattern 2: SWI-Only (Web Application)**

```prolog
% Uses SWI-specific features
:- use_module(library(http/thread_httpd)).
:- use_module(library(http/http_dispatch)).

% Generate for SWI deployment
?- generate_prolog_script([start_server/0],
                         [dialect(swi)],
                         Code).
```

**Pattern 3: Multi-Target**

```prolog
% Generate both versions
generate_both(Predicates, BaseName) :-
    % SWI version (for development)
    generate_prolog_script(Predicates,
                          [dialect(swi)],
                          SwiCode),
    atom_concat(BaseName, '_swi.pl', SwiPath),
    write_prolog_script(SwiCode, SwiPath),

    % GNU version (for production)
    generate_prolog_script(Predicates,
                          [dialect(gnu), compile(true)],
                          GnuCode),
    atom_concat(BaseName, '_gnu.pl', GnuPath),
    write_prolog_script(GnuCode, GnuPath,
                       [dialect(gnu), compile(true)]).
```

## Compatibility Considerations

### Portable Subset

To maximize portability across dialects, stick to:

✅ **Core Prolog**:
- Basic predicates (append, member, etc.)
- Arithmetic (is, =:=, >, <, etc.)
- Control (!, ->, if-then-else)
- Meta-predicates (findall, bagof, setof)

✅ **Standard I/O**:
- read, write, format
- open, close, read_string
- File operations

❌ **Avoid Dialect-Specific**:
- SWI: tabling, threads, HTTP
- GNU: gprolog-specific built-ins
- Either: dialect-specific libraries

### UnifyWeaver's Validation

UnifyWeaver can detect incompatibilities:

```prolog
validate_for_dialect(gnu, [my_predicate/2], Issues).
% Issues = [
%     unsupported_feature(tabling, 'GNU Prolog does not support tabling'),
%     missing_library(http, 'GNU Prolog lacks HTTP support')
% ]
```

## What's Next?

Chapter 4 shows practical usage of the Prolog target, starting with basic examples and building up to production scenarios.

---

**Key Takeaways:**
- SWI-Prolog: Rich features, ideal for development
- GNU Prolog: Compiled binaries, ideal for deployment
- Choose based on: deployment needs, library requirements, performance
- UnifyWeaver enables cross-dialect development
- Initialization directives differ between dialects and modes
- Validation helps catch incompatibilities early

---

## Navigation

**←** [Previous: Chapter 2: Architecture Overview](02_architecture) | [📖 Book 11: Prolog Target](./) | [Next: Chapter 7: The Firewall System →](07_firewall_integration)

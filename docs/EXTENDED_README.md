<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2025 John William Creighton (@s243a)
-->

# UnifyWeaver Extended Documentation

This document provides comprehensive documentation for UnifyWeaver, including detailed examples, tutorials, and advanced usage patterns.

**For a quick overview, see the [main README](../README.md).**

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Core Features](#core-features)
4. [Advanced Recursion Patterns](#advanced-recursion-patterns)
5. [Data Source Plugin System](#data-source-plugin-system)
6. [Firewall and Security](#firewall-and-security)
7. [PowerShell Target](#powershell-target)
8. [C# Target Family](#c-target-family)
9. [JVM Targets](#jvm-targets)
10. [Native Targets (C/C++)](#native-targets-cc)
11. [Client-Server Architecture](#client-server-architecture)
12. [Complete Examples](#complete-examples)
13. [Architecture Deep Dive](#architecture-deep-dive)
14. [Testing Guide](#testing-guide)
15. [Troubleshooting](#troubleshooting)

---

## Introduction

UnifyWeaver is a Prolog-to-Bash compiler that transforms declarative logic programs into efficient streaming bash scripts. It specializes in compiling data relationships and queries into executable bash code with optimized handling of transitive closures and advanced recursion patterns.

### Why UnifyWeaver?

- **Declarative to Imperative** - Write logic once in Prolog, compile to bash for production
- **Performance Optimized** - Automatic BFS optimization, memoization, and loop generation
- **Stream Processing** - Memory-efficient compilation using bash pipes
- **Production Ready** - Data source plugins, firewall security, comprehensive testing

### Use Cases

1. **ETL Pipelines** - CSV/JSON processing with SQLite storage
2. **Graph Queries** - Family trees, dependency graphs, network analysis
3. **API Integration** - REST API consumption with caching
4. **Data Transformation** - Stream processing with filters and aggregation

---

## Installation and Setup

### Requirements

- **SWI-Prolog** 8.0 or higher
- **Bash** 4.0+ (for associative arrays)
- **Optional Tools**:
  - `awk` for text processing and field extraction
  - `jq` for JSON processing
  - `curl` or `wget` for HTTP sources
  - `python3` for Python integration
  - `sqlite3` for database operations

### Installation

```bash
git clone https://github.com/s243a/UnifyWeaver.git
cd UnifyWeaver
```

### Quick Start with Test Environment

UnifyWeaver includes a convenient test environment with auto-discovery:

**Linux/WSL:**
```bash
cd scripts/testing
./init_testing.sh
```

**Windows PowerShell:**
```powershell
cd scripts\testing
.\Init-TestEnvironment.ps1
```

**In SWI-Prolog:**
```prolog
?- test_all.           % Run all tests
?- test_stream.        % Test stream compilation
?- test_recursive.     % Test basic recursion
?- test_advanced.      % Test advanced recursion patterns
?- test_constraints.   % Test constraint system
```

### Manual Setup

```prolog
?- use_module(unifyweaver(core/recursive_compiler)).
?- test_recursive_compiler.
```

Generated scripts appear in the `output/` directory:
```bash
cd output
bash test_recursive.sh
```

---

## Core Features

### Stream-Based Compilation

UnifyWeaver compiles Prolog predicates to bash functions that work with streams:

```prolog
% Define facts
parent(alice, bob).
parent(bob, charlie).
parent(alice, diana).

% Compile to bash
?- compile_recursive(parent/2, [], BashCode).
?- write_bash_file('parent.sh', BashCode).
```

Generated bash provides multiple interfaces:

```bash
source parent.sh

# Stream all results
parent_all alice              # bob\ndiana

# Check specific relationship
parent_check alice bob && echo "Yes"  # Yes

# Search patterns
parent_search alice ""        # All children of alice
parent_search "" charlie      # All parents of charlie
```

### BFS Optimization for Transitive Closures

UnifyWeaver automatically detects transitive closure patterns and optimizes them to breadth-first search:

```prolog
% Transitive closure pattern
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Compile with automatic BFS optimization
?- compile_recursive(ancestor/2, [], BashCode).
```

The generated bash:
- Uses work queues instead of recursion
- Prevents cycles with visited tracking
- Eliminates duplicate results
- Runs in O(V+E) time

### Cycle Detection

Handles cyclic graphs correctly:

```prolog
% Cyclic graph
connected(a, b).
connected(b, c).
connected(c, a).  % Cycle!

% Transitive closure
reachable(X, Y) :- connected(X, Y).
reachable(X, Z) :- connected(X, Y), reachable(Y, Z).

% Generated bash handles cycles without infinite loops
```

### Constraint System

UnifyWeaver detects constraints and optimizes accordingly:

**Unique Constraint** - When only one result is expected:
```prolog
% Compile with unique constraint
?- compile_recursive(factorial/2, [unique(true)], BashCode).
```

Generated bash includes early exit optimization:
```bash
# Stops after finding first result
if [[ "${memo[$key]}" ]]; then
    echo "${memo[$key]}"
    return 0  # Early exit
fi
```

**Unordered Constraint** - When result order doesn't matter:
```prolog
?- compile_recursive(ancestor/2, [unordered(true)], BashCode).
```

Enables optimizations like parallel processing and set-based deduplication.

---

## Advanced Recursion Patterns

UnifyWeaver automatically detects and optimizes four advanced recursion patterns:

### 1. Tail Recursion → Iterative Loops

**Pattern:** Recursive call is the last operation (tail position).

```prolog
% Count list items with accumulator
count_items([], Acc, Acc).
count_items([_|T], Acc, N) :-
    Acc1 is Acc + 1,
    count_items(T, Acc1, N).

% Compile with unique constraint
?- compile_advanced_recursive(count_items/3, [unique(true)], BashCode).
```

**Generated bash (simplified):**
```bash
count_items() {
    local list="$1"
    local acc="$2"

    while true; do
        if [[ "$list" == "[]" ]]; then
            echo "$acc"
            return 0
        fi
        # Extract head and tail
        acc=$((acc + 1))
        list="$tail"
    done
}
```

**Benefits:**
- O(1) space instead of O(n) stack
- No recursion depth limits
- Faster execution

### 2. Linear Recursion → Memoization

**Pattern:** Single recursive call, independent computation.

```prolog
% Fibonacci sequence
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Compile with linear recursion
?- compile_advanced_recursive(fib/2, [], BashCode).
```

**Generated bash (simplified):**
```bash
declare -A fib_memo

fib() {
    local n="$1"
    local key="$n"

    # Check memo table
    if [[ "${fib_memo[$key]}" ]]; then
        echo "${fib_memo[$key]}"
        return 0
    fi

    # Base cases
    if [[ "$n" == "0" ]]; then
        fib_memo[$key]="0"
        echo "0"
        return 0
    fi

    # Recursive computation with memoization
    # ... (implementation details)
}
```

**Benefits:**
- Avoids redundant computation
- Dynamic programming optimization
- Works for factorial, list length, tree height, etc.

### 3. Tree Recursion → Structural Processing

**Pattern:** Multiple recursive calls on parts of a structure.

```prolog
% Binary tree sum using list representation: [value, [left], [right]]
tree_sum([], 0).
tree_sum([V, L, R], Sum) :-
    tree_sum(L, LS),
    tree_sum(R, RS),
    Sum is V + LS + RS.

% Compile with tree recursion pattern
?- compile_advanced_recursive(tree_sum/2, [], BashCode).
```

**Usage:**
```bash
source tree_sum.sh
tree_sum "[10,[5,[],[3,[],[]]],[7,[],[]]]"  # Returns: 25
```

**Tree Representation:**
```
       10
      /  \
     5    7
      \
       3

Input: "[10,[5,[],[3,[],[]]],[7,[],[]]]"
Result: 10 + 5 + 3 + 7 = 25
```

### 4. Mutual Recursion → Shared Memoization

**Pattern:** Predicates that call each other in cycles.

```prolog
% Even/odd mutual recursion
is_even(0).
is_even(N) :- N > 0, N1 is N - 1, is_odd(N1).

is_odd(1).
is_odd(N) :- N > 1, N1 is N - 1, is_even(N1).

% Compile predicate group together
?- compile_predicate_group([is_even/1, is_odd/1], [unique(true)], BashCode).
```

**Generated bash features:**
- Shared memo table across both predicates
- Mutual calls between functions
- Cycle detection

**Benefits:**
- Handles complex interdependencies
- Automatic SCC (Strongly Connected Components) detection
- Shared optimization across predicates

---

## Data Source Plugin System

UnifyWeaver v0.0.2 introduces a powerful plugin system for integrating external data sources.

### Architecture

```
┌─────────────────────────────────────────────┐
│  Prolog Source Declaration (source/3)      │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│  Plugin Registry (Auto-Discovery)          │
│  - CSV Plugin                              │
│  - AWK Plugin                              │
│  - JSON Plugin                             │
│  - HTTP Plugin                             │
│  - Python Plugin                           │
│  - C# Plugin (LiteDB)                      │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│  Compilation (compile_dynamic_source/3)    │
│  - Validate options                        │
│  - Generate bash code                      │
│  - Apply templates                         │
└────────────────┬────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────┐
│  Generated Bash Function                   │
│  - Error handling                          │
│  - Streaming output                        │
│  - Resource cleanup                        │
└─────────────────────────────────────────────┘
```

### CSV/TSV Plugin

Process CSV and TSV files with automatic header detection:

**Auto-detect headers:**
```prolog
:- source(csv, users, [
    csv_file('data/users.csv'),
    has_header(true)
]).
```

**Manual column specification:**
```prolog
:- source(csv, logs, [
    csv_file('data/access.log'),
    delimiter('\t'),
    columns([timestamp, ip, method, path, status])
]).
```

**Generated bash:**
```bash
users() {
    awk -F',' 'NR==1 {next} {print $1 ":" $2 ":" $3}' data/users.csv
}
```

**Usage:**
```prolog
?- users(Name, Age, City).
Name = alice, Age = 25, City = nyc.
```

### AWK Plugin

Process text files with AWK pattern matching and field extraction:

**Pattern matching with field extraction:**
```prolog
:- source(awk, high_scorers, [
    awk_program('$3 > 100 { print $1, $2, $3 }'),
    input_file('scores.txt'),
    field_separator(' ')
]).
```

**Complex AWK scripts:**
```prolog
:- source(awk, sales_summary, [
    awk_program('BEGIN { total = 0 } { total += $2 } END { print total }'),
    input_file('sales.txt')
]).
```

**Generated bash:**
```bash
high_scorers() {
    awk -F' ' '$3 > 100 { print $1, $2, $3 }' scores.txt
}
```

**Usage:**
```prolog
?- high_scorers(Name, Score, Rank).
Name = alice, Score = 150, Rank = 1.
```

### JSON Plugin

Process JSON data with jq filters:

**Parse JSON with filters:**
```prolog
:- source(json, extract_names, [
    jq_filter('.users[] | {name: .name, email: .email} | @tsv'),
    json_stdin(true),
    raw_output(true)
]).
```

**Process JSON files:**
```prolog
:- source(json, config_values, [
    json_file('config.json'),
    columns(['database.host', 'database.port', 'database.name'])
]).

?- config_values(Host, Port, Name).
```

Returning full JSON objects instead of individual fields:

```prolog
:- source(json, raw_products, [
    json_file('test_data/test_products.json'),
    arity(1),
    return_object(true),
    type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json')
]).

?- raw_products(ProductJson).
```

Generating a typed record via `schema/1`:

```prolog
:- source(json, product_rows, [
    json_file('test_data/test_products.json'),
    schema([
        field(id, 'id', string),
        field(name, 'name', string),
        field(price, 'price', double)
    ]),
    record_type('ProductRecord')
]).

?- product_rows(Row).
% Row = ProductRecord { Id = P001, Name = Laptop, Price = 999 }
```

Selecting via JSONPath (using wildcards and recursive descent):

```prolog
:- source(json, order_first_items, [
    json_file('test_data/test_orders.json'),
    columns([
        jsonpath('$.order.customer.name'),
        jsonpath('$.items[*].product')
    ])
]).

?- order_first_items(Customer, Product).
% Customer = Alice,  Product = Laptop
```

Combining schema + nested records:

```prolog
:- source(json, order_summaries, [
    json_file('test_data/test_orders.json'),
    schema([
        field(order, 'order', record('OrderRecord', [
            field(id, 'id', string),
            field(customer, 'customer.name', string)
        ])),
        field(first_item, 'items[0]', record('LineItemRecord', [
            field(product, 'product', string),
            field(total, 'total', double)
        ]))
    ]),
    record_type('OrderSummaryRecord')
]).

?- order_summaries(Row).
% Row = OrderSummaryRecord { Order = OrderRecord { Id = SO1, Customer = Alice },
%                            FirstItem = LineItemRecord { Product = Laptop, Total = 1200 } }
```

Streaming JSON Lines (`record_format(jsonl)`) with null-handling policy:

```prolog
:- source(json, order_second_items, [
    json_file('test_data/test_orders.jsonl'),
    record_format(jsonl),
    columns([
        jsonpath('$.order.customer.name'),
        jsonpath('$.items[1].product')
    ]),
    null_policy(default('N/A'))
]).

?- order_second_items(Customer, SecondProduct).
% Customer = Alice, SecondProduct = Mouse
% Customer = Bob,   SecondProduct = 'N/A'
% Customer = Charlie, SecondProduct = 'N/A'
```

`null_policy(fail|skip|default(Value))` controls how missing fields behave: fail immediately, skip the row, or substitute the provided default string.

**Generated bash:**
```bash
extract_names() {
    jq -r '.users[] | {name: .name, email: .email} | @tsv'
}
```

### YAML Plugin

Process YAML data using Python (PyYAML):

**Read YAML files with filters:**
```prolog
:- source(yaml, config_users, [
    yaml_filter('data["users"]'),
    yaml_file('config.yaml')
]).
```

**Process YAML from stdin:**
```prolog
:- source(yaml, stream_data, [
    yaml_stdin(true)
]).
```

**Generated bash:**
```bash
config_users() {
    python3 << 'PYTHON'
    import yaml
    # ... (embedded python script)
PYTHON
}
```

### XML Plugin

Process XML data using Python:

**Parse XML with Python:**
```prolog
:- data_source_driver(python).
:- data_source_work_fn(sum_prices).

:- data_source_schema([
    string(name),
    number(price)
]).

:- data_source_prolog_fn(xml_prolog_data_source).

data_source_definition(xml_data_source, _{
    driver: python,
    work_fn: sum_prices,
    schema: [string(name), number(price)],
    prolog_fn: xml_prolog_data_source
}).
```

### HTTP Plugin

Fetch data from REST APIs with caching:

**GET requests with caching:**
```prolog
:- source(http, github_repos, [
    url('https://api.github.com/users/octocat/repos'),
    headers(['User-Agent: UnifyWeaver/0.0.2']),
    cache_duration(3600),  % 1 hour cache
    cache_file('cache/github_repos.json')
]).
```

**POST requests:**
```prolog
:- source(http, webhook, [
    url('https://hooks.example.com/notify'),
    method(post),
    headers(['Content-Type: application/json']),
    post_data('{"status": "completed"}')
]).
```

**Generated bash:**
```bash
github_repos() {
    if [[ -f cache/github_repos.json ]]; then
        # Check cache age
        # ... (cache logic)
    fi
    curl -H 'User-Agent: UnifyWeaver/0.0.2' \
         'https://api.github.com/users/octocat/repos'
}
```

### Python Plugin

Execute Python code with SQLite integration:

**Inline Python:**
```prolog
:- source(python, store_data, [
    python_inline('
import sqlite3
import sys
conn = sqlite3.connect("app.db")
conn.execute("CREATE TABLE IF NOT EXISTS results (key, value)")
for line in sys.stdin:
    key, value = line.strip().split(":")
    conn.execute("INSERT INTO results VALUES (?, ?)", (key, value))
conn.commit()
print("Data stored successfully")
'),
    timeout(30)
]).
```

**Direct SQLite queries:**
```prolog
:- source(python, get_users, [
    sqlite_query('SELECT name, age FROM users WHERE active = 1'),
    database('app.db')
]).
```

**Generated bash:**
```bash
store_data() {
    python3 << 'PYTHON_EOF'
import sqlite3
import sys
conn = sqlite3.connect("app.db")
# ... (full script)
PYTHON_EOF
}
```

### SQLite Plugin (Native)

Query SQLite databases directly using the `sqlite3` command-line tool (faster than Python wrapper for simple queries):

```prolog
:- source(sqlite, high_value_orders, [
    sqlite_file('data/orders.db'),
    query('SELECT id, total FROM orders WHERE total > 1000'),
    output_format(tsv)
]).
```

**Generated bash:**
```bash
high_value_orders() {
    sqlite3 -separator '	' -noheader "data/orders.db" "SELECT id, total FROM orders WHERE total > 1000"
}
```

**Safe Parameter Binding (v1.1):**

To prevent SQL injection when using dynamic inputs, use `parameters/1`. This automatically switches to a safer Python-based execution mode:

```prolog
:- source(sqlite, user_by_id, [
    sqlite_file('data/users.db'),
    query('SELECT name, email FROM users WHERE id = ?'),
    parameters(['$1'])  % Binds first script argument to ?
]).
```

### C# Plugin (LiteDB Integration)

UnifyWeaver supports C# data sources with automatic compilation and LiteDB integration.

**LiteDB Inserter:**
```prolog
:- source(dotnet, load_products, [
    arity(0),
    target(powershell),
    csharp_inline('
using LiteDB;
// ... C# code ...
'),
    references(['lib/LiteDB.dll'])
]).
```

**See [C# Compilation Guide](DOTNET_COMPILATION.md) for full details.**

### Complete ETL Pipeline Example

```prolog
% Configure firewall for data processing
:- assertz(firewall:firewall_default([
    services([awk, python3, curl, jq]),
    network_access(allowed),
    network_hosts(['*.github.com', '*.typicode.com']),
    python_modules([sys, json, sqlite3, csv]),
    file_read_patterns(['data/*', 'config/*']),
    cache_dirs(['/tmp/*', 'cache/*'])
])).

% Define data sources
:- source(http, api_data, [
    url('https://api.github.com/users/octocat/repos')
]).

:- source(json, parse_repos, [
    jq_filter('.[] | {name: .name, stars: .stargazers_count} | @tsv'),
    json_stdin(true)
]).

:- source(python, store_repos, [
    python_inline('
import sqlite3
import sys
conn = sqlite3.connect("repos.db")
conn.execute("CREATE TABLE IF NOT EXISTS repos (name, stars)")
for line in sys.stdin:
    name, stars = line.strip().split("\t")
    conn.execute("INSERT INTO repos VALUES (?, ?)", (name, stars))
conn.commit()
')
]).

% Pipeline: API → JSON → SQLite
etl_pipeline :-
    api_data | parse_repos | store_repos.
```

**Run the pipeline:**
```bash
swipl -g etl_pipeline -t halt my_etl.pl
```

---

## Firewall and Security

UnifyWeaver includes a comprehensive firewall system for controlling external tool usage, network access, and file operations.

### Firewall Philosophy

The firewall operates at three levels:

1. **Global/Default Level** - System-wide preferences
2. **Firewall Policy Level** - Policy-specific rules
3. **Compilation Options Level** - Per-predicate overrides

**Priority:** Compilation Options > Firewall Policy > Global Default

### Network Access Control

Control which URLs and domains can be accessed:

**Deny all network access:**
```prolog
load_firewall_policy(no_network).
```

**Whitelist specific domains:**
```prolog
load_firewall_policy(whitelist_domains(['example.com', 'trusted.org'])).
```

**Blacklist specific domains:**
```prolog
assertz(denied_domain('malicious.com')).
```

**URL pattern matching:**
```prolog
% Block admin endpoints
assertz(denied_url_pattern('admin')).

% Check URL access
?- check_url_access('https://example.com/admin/users', Result).
Result = deny(url_pattern_denied(admin)).
```

### Domain Pattern Matching

Supports exact match, subdomain match, and wildcards:

```prolog
% Exact match
domain_matches_pattern('example.com', 'example.com').  % true

% Subdomain match
domain_matches_pattern('api.example.com', 'example.com').  % true

% Wildcard match
domain_matches_pattern('api.example.com', '*.example.com').  % true
```

### Service Control

Control which external tools can be used:

```prolog
% Allow specific services
assertz(allowed_service(bash, executable(awk))).
assertz(allowed_service(powershell, cmdlet(import_csv))).

% Deny services
assertz(denied_service(powershell, executable(bash))).  % Pure PowerShell
```

### Python Module Whitelisting

Control which Python modules can be imported:

```prolog
% Allow only specific modules
assertz(allowed_python_module(sys)).
assertz(allowed_python_module(json)).
assertz(allowed_python_module(sqlite3)).

% Deny dangerous modules
assertz(denied_python_module(os)).
assertz(denied_python_module(subprocess)).
```

### File Access Patterns

Control file read/write permissions:

```prolog
% Allow reading from data directories
assertz(allowed_file_read('data/*')).
assertz(allowed_file_read('config/*')).

% Allow writing to output directories
assertz(allowed_file_write('output/*')).
assertz(allowed_file_write('/tmp/*')).
```

### Complete Firewall Example

```prolog
% Load strict security policy
load_firewall_policy(strict_security).

% Configure allowed services
assertz(allowed_service(bash, executable(awk))).
assertz(allowed_service(bash, executable(grep))).

% Network access
assertz(network_access_policy(whitelist)).
assertz(allowed_domain('api.internal.company.com')).

% Python restrictions
assertz(allowed_python_module(sys)).
assertz(allowed_python_module(json)).

% File access
assertz(allowed_file_read('data/*')).
assertz(allowed_file_write('output/*')).

% Check if operation is allowed
?- check_firewall(service(bash, executable(curl)), Result).
Result = deny(service_not_allowed).

?- check_url_access('https://api.internal.company.com/data', Result).
Result = allow.
```

For complete firewall documentation, see [FIREWALL_GUIDE.md](FIREWALL_GUIDE.md).

---

## PowerShell Target

UnifyWeaver supports compiling Prolog predicates to PowerShell with two modes:

### Pure PowerShell Mode

Uses only PowerShell cmdlets (no bash/AWK dependencies):

```prolog
% Compile CSV source to pure PowerShell
compile_to_powershell(users/3, [
    source_type(csv),
    csv_file('users.csv'),
    powershell_mode(pure)
], Code).
```

**Generated PowerShell:**
```powershell
function users {
    Import-Csv -Path 'users.csv' | ForEach-Object {
        "$($_.Name):$($_.Age):$($_.City)"
    }
}
```

### BaaS Mode (Bash-as-a-Service)

Uses bash/AWK for performance when available:

```prolog
compile_to_powershell(users/3, [
    source_type(csv),
    csv_file('users.csv'),
    powershell_mode(baas)
], Code).
```

**Generated PowerShell:**
```powershell
function users {
    bash -c "awk -F',' 'NR>1 {print \$1\":\"\$2\":\"\$3}' users.csv"
}
```

### Auto Mode

Automatically selects pure or BaaS based on data source:

```prolog
compile_to_powershell(users/3, [
    source_type(csv),
    csv_file('users.csv'),
    powershell_mode(auto)  % Chooses pure for CSV
], Code).
```

For complete PowerShell documentation, see:
- [POWERSHELL_TARGET.md](POWERSHELL_TARGET.md)
- [POWERSHELL_PURE_IMPLEMENTATION.md](POWERSHELL_PURE_IMPLEMENTATION.md)
- [POWERSHELL_PURE_VS_BAAS.md](POWERSHELL_PURE_VS_BAAS.md)

---

## C# Target Family

UnifyWeaver provides a robust C# compilation system that integrates seamlessly with PowerShell.

### Features

- **External Compilation**: Uses `dotnet build` for robust dependency management and isolation.
- **Pre-Compilation**: Fallback to `Add-Type` for simple scripts.
- **File Locking Solution**: Unique build directories prevent DLL locking issues.
- **LiteDB Support**: Built-in integration for NoSQL document storage.
- **Dynamic Sources (CSV/JSON/XML)**:
  - `DelimitedTextReader` for CSV/TSV (NUL- or LF-delimited)
  - `JsonStreamReader` for JSON/JSONL with schema/type hints and null policy
  - `XmlStreamReader` for NUL- or LF-delimited XML fragments; matches both prefixes and fully-qualified names and projects elements/attributes into a dictionary per record.

**See [DOTNET_COMPILATION.md](DOTNET_COMPILATION.md) for the complete guide.**

---

## JVM Targets

UnifyWeaver supports multiple JVM-based languages with full transitive closure and fact export:

### Supported Languages

| Target | Transitive Closure | Fact Export | Key Features |
|--------|-------------------|-------------|--------------|
| Java | ✅ | ✅ | `HashSet`, `ArrayList`, Streams |
| Kotlin | ✅ | ✅ | `sequence { yield() }`, null safety |
| Scala | ✅ | ✅ | `LazyList`, pattern matching |
| Clojure | ✅ | ✅ | `loop/recur`, lazy sequences |
| Jython | ✅ | ✅ | Python 2.7 on JVM |

### Quick Start

```prolog
% Transitive closure (ancestor query)
?- compile_recursive(ancestor/2, [target(java)], Code).
?- compile_recursive(ancestor/2, [target(kotlin)], Code).
?- compile_recursive(ancestor/2, [target(scala)], Code).

% Export facts as static data
?- compile_facts_to_java(parent, 2, Code).
?- compile_facts_to_kotlin(parent, 2, Code).
```

### Implementation Details

All JVM targets use BFS (Breadth-First Search) for transitive closure:
- **Java**: `HashSet<String>` for visited, `ArrayDeque<String>` for queue
- **Kotlin**: `mutableSetOf()`, `ArrayDeque()`
- **Scala**: `mutable.Set`, `mutable.Queue`
- **Clojure**: `loop/recur` with persistent data structures

**See the [education submodule](education/) for detailed tutorials per language.**

---

## Native Targets (C/C++)

UnifyWeaver compiles Prolog to native C and C++ with full transitive closure support:

### Features

| Target | Transitive Closure | Fact Export | Key Features |
|--------|-------------------|-------------|--------------|
| C | ✅ | ✅ | Manual memory, adjacency list |
| C++ | ✅ | ✅ | STL containers, `std::optional` |

### Quick Start

```prolog
% Transitive closure
?- compile_recursive(ancestor/2, [target(c)], Code).
?- compile_recursive(ancestor/2, [target(cpp)], Code).

% Export facts
?- compile_facts_to_c(parent, 2, Code).
?- compile_facts_to_cpp(parent, 2, Code).
```

### C Implementation

```c
/* BFS with manual adjacency list */
typedef struct Edge { char* to; struct Edge* next; } Edge;
typedef struct { char* from; Edge* edges; } Node;

static void find_all(const char* start) {
    char* queue[MAX_NODES];
    int visited[MAX_NODES] = {0};
    // BFS implementation...
}
```

### C++ Implementation

```cpp
class AncestorQuery {
    std::unordered_map<std::string, std::vector<std::string>> baseRelation;
public:
    std::vector<std::string> findAll(const std::string& start) {
        std::unordered_set<std::string> visited;
        std::queue<std::string> queue;
        // BFS implementation...
    }
};
```

---

## Client-Server Architecture

UnifyWeaver v0.9 introduces a comprehensive client-server architecture for building production-grade distributed services. The architecture spans 8 phases, from simple in-process services to fully distributed systems with observability.

### Overview

The client-server system uses a unified `service/3` predicate that separates business logic from transport and infrastructure concerns:

```prolog
service(ServiceName, Options, Operations).
```

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Layer                               │
│              service(name, options, operations)                      │
├─────────────────────────────────────────────────────────────────────┤
│                      Service Mesh Layer                              │
│         Load Balancing │ Circuit Breaker │ Retry │ Rate Limit       │
├─────────────────────────────────────────────────────────────────────┤
│                      Discovery Layer                                 │
│              Consul │ etcd │ DNS │ Kubernetes                        │
├─────────────────────────────────────────────────────────────────────┤
│                      Transport Layer                                 │
│           In-Process │ Unix Socket │ TCP │ HTTP                      │
├─────────────────────────────────────────────────────────────────────┤
│                      Observability Layer                             │
│              OpenTelemetry │ Jaeger │ Zipkin │ Prometheus            │
└─────────────────────────────────────────────────────────────────────┘
```

### Phase Summary

| Phase | Feature | Description |
|-------|---------|-------------|
| 1 | In-Process Services | Direct function calls within same process |
| 2 | Unix Socket Services | IPC for local inter-process communication |
| 3 | Network Services | TCP/HTTP for distributed deployment |
| 4 | Service Mesh | Load balancing, circuit breaker, retry |
| 5 | Polyglot Services | Cross-language service communication |
| 6 | Distributed Services | Sharding, replication, consistency |
| 7 | Service Discovery | Runtime service registration and lookup |
| 8 | Distributed Tracing | Request tracking across service boundaries |

### Quick Start Example

```prolog
% Define a user service with full production features
service(user_service, [
    % Transport
    transport(http(8080)),

    % Resilience
    load_balancing(round_robin),
    circuit_breaker(true),
    retry_strategy(exponential),

    % Distributed
    sharding(consistent_hash),
    replication(3),
    consistency(quorum),

    % Discovery & Tracing
    discovery_backend(consul),
    health_check(http('/health', 30000)),
    tracing(true),
    trace_exporter(otlp)
], [
    receive(Request),
    state_get(users, Users),
    respond(Response)
]).
```

### Service Options Reference

#### Transport Options
| Option | Values | Description |
|--------|--------|-------------|
| `transport(T)` | `in_process`, `unix_socket(Path)`, `tcp(Port)`, `http(Port)` | Service transport |

#### Resilience Options
| Option | Values | Description |
|--------|--------|-------------|
| `load_balancing(S)` | `round_robin`, `random`, `least_connections`, `weighted`, `ip_hash` | Load balancing strategy |
| `circuit_breaker(B)` | `true`, `false` | Enable circuit breaker |
| `retry_strategy(S)` | `fixed`, `linear`, `exponential` | Retry backoff strategy |

#### Distributed Options
| Option | Values | Description |
|--------|--------|-------------|
| `sharding(S)` | `hash`, `range`, `consistent_hash`, `geographic` | Data partitioning |
| `replication(N)` | Integer | Number of replicas |
| `consistency(L)` | `eventual`, `strong`, `quorum`, `causal` | Consistency level |

#### Discovery & Tracing Options
| Option | Values | Description |
|--------|--------|-------------|
| `discovery_backend(B)` | `consul`, `etcd`, `dns`, `kubernetes` | Service discovery backend |
| `health_check(C)` | `http(Path, Interval)`, `tcp(Port, Interval)` | Health check configuration |
| `tracing(B)` | `true`, `false` | Enable distributed tracing |
| `trace_exporter(E)` | `otlp`, `jaeger`, `zipkin`, `console` | Trace export destination |

### Target Language Support

The client-server architecture generates idiomatic code for:

| Target | Transport | Mesh | Polyglot | Distributed | Discovery | Tracing |
|--------|-----------|------|----------|-------------|-----------|---------|
| Python | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Go | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Rust | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Documentation

For comprehensive documentation, see the **Education Project** Book 7: Cross-Target Glue:

- **Chapter 11a**: Service Architecture (Phases 1-3)
- **Chapter 12a**: Service Mesh (Phase 4)
- **Chapter 12b**: Polyglot and Distributed Services (Phases 5-6)
- **Chapter 12c**: Service Discovery and Tracing (Phases 7-8)

Design documentation is also available at [CLIENT_SERVER_DESIGN.md](CLIENT_SERVER_DESIGN.md).

---

## Complete Examples

### Example 1: Family Tree with Transitive Queries

```prolog
% facts.pl
parent(alice, bob).
parent(alice, charlie).
parent(bob, dave).
parent(charlie, eve).
parent(dave, frank).

% queries.pl
:- use_module(unifyweaver(core/recursive_compiler)).

% Compile parent (facts)
compile_parent :-
    compile_recursive(parent/2, [], ParentCode),
    write_bash_file('parent.sh', ParentCode).

% Compile ancestor (transitive closure)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

compile_ancestor :-
    compile_recursive(ancestor/2, [], AncestorCode),
    write_bash_file('ancestor.sh', AncestorCode).

% Compile both
main :- compile_parent, compile_ancestor.
```

**Usage:**
```bash
swipl -g main -t halt queries.pl
source parent.sh
source ancestor.sh

# Find all descendants of alice
ancestor_all alice

# Output:
# bob
# charlie
# dave
# eve
# frank
```

### Example 2: ETL Pipeline with Multiple Sources

See the complete working example:

```bash
cd scripts/testing/test_env5
swipl -g main -t halt examples/pipeline_demo.pl
```

This demonstrates:
- CSV source with 4 users
- Streaming all records
- Filtering by role (developers)
- Aggregating by department
- Writing results to file

**Generated files:**
- `output/users.sh` - Compiled bash functions
- `output/run_pipeline.sh` - Pipeline script
- `output/pipeline_results.txt` - Results

### Example 3: Graph Reachability with Cycles

```prolog
% Graph with cycle
edge(a, b).
edge(b, c).
edge(c, d).
edge(d, b).  % Cycle: b → c → d → b

% Reachable nodes (transitive closure)
reachable(X, Y) :- edge(X, Y).
reachable(X, Z) :- edge(X, Y), reachable(Y, Z).

% Compile
?- compile_recursive(reachable/2, [], Code).
?- write_bash_file('reachable.sh', Code).
```

**Usage:**
```bash
source edge.sh
source reachable.sh

# Find all nodes reachable from 'a'
reachable_all a

# Output (no duplicates, no infinite loop):
# b
# c
# d
```

### Example 4: Fibonacci with Memoization

```prolog
fib(0, 0).
fib(1, 1).
fib(N, F) :-
    N > 1,
    N1 is N - 1,
    N2 is N - 2,
    fib(N1, F1),
    fib(N2, F2),
    F is F1 + F2.

% Compile with linear recursion optimization
?- compile_advanced_recursive(fib/2, [unique(true)], Code).
?- write_bash_file('fib.sh', Code).
```

**Usage:**
```bash
source fib.sh

# Calculate Fibonacci number
fib 10

# Output:
# 55
```

### Example 5: XML Data Source

See the complete working example in the playbook: `playbooks/xml_data_source_playbook.md`.

This demonstrates:
- Defining a data source using Python to parse XML.
- Extracting data from the XML.
- Executing the data source.

### Example 6: WASM Visualization (Graph and Curves)

UnifyWeaver compiles Prolog to WebAssembly via LLVM for browser-based visualization.

**Graph Visualization** (`examples/wasm-graph/`):
```prolog
% Compile graph operations to WASM
compile_wasm_string_module(
    [func(ancestor, 2, transitive_closure)],
    [module_name(family_graph)],
    LLVMCode).
```
- Cytoscape.js for network graph visualization
- TypeScript bindings with string support
- Interactive edge addition and graph layout

**Curve Plotting** (`examples/curve-plot/`):
```prolog
% Define mathematical curves
generate_curve_wasm([
    curve_def(wave, sine),
    curve_def(parabola, quadratic)
], LLVMCode).

% Evaluate in Prolog
?- evaluate_curve(sine(1, 2, 0), 3.14, Y).
Y = 0.0.
```
- Chart.js for mathematical curve visualization
- Configurable parameters (amplitude, frequency, coefficients)
- Multiple curve overlays with different colors

**Pipeline:**
```
Prolog → LLVM IR → WebAssembly → TypeScript → Browser
```

**Build:**
```bash
cd examples/curve-plot
swipl curve_module.pl                           # Generate LLVM IR
llc -march=wasm32 -filetype=obj curve_plot.ll   # Compile to WASM object
wasm-ld --no-entry --export-all curve_plot.o -o curve_plot.wasm
python3 -m http.server 8080                     # Serve demo
```

See [LLVM_TARGET.md](LLVM_TARGET.md) for full API documentation.

---

## Architecture Deep Dive

### Module Structure

```
src/unifyweaver/
├── core/
│   ├── template_system.pl          # Mustache-style template engine
│   ├── stream_compiler.pl          # Non-recursive predicate compilation
│   ├── recursive_compiler.pl       # Basic recursion with BFS optimization
│   ├── constraint_analyzer.pl      # Unique/ordering constraint detection
│   ├── firewall_v2.pl             # Security and policy enforcement
│   ├── preferences.pl              # Layered configuration
│   ├── tool_detection.pl           # Cross-platform tool availability
│   └── advanced/
│       ├── advanced_recursive_compiler.pl  # Orchestrator
│       ├── call_graph.pl                   # Dependency graph builder
│       ├── scc_detection.pl                # Tarjan's algorithm
│       ├── pattern_matchers.pl             # Pattern detection
│       ├── tail_recursion.pl               # Tail → loop compiler
│       ├── linear_recursion.pl             # Linear → memo compiler
│       ├── tree_recursion.pl               # Tree → structural compiler
│       └── mutual_recursion.pl             # Mutual → shared memo compiler
└── plugins/
    ├── plugin_registry.pl          # Auto-discovery system
    └── sources/
        ├── csv_source.pl           # CSV/TSV plugin
        ├── json_source.pl          # JSON/jq plugin
        ├── http_source.pl          # HTTP/REST plugin
        └── python_source.pl        # Python/SQLite plugin
```

### Compilation Pipeline

```
┌─────────────────────────────────────┐
│  1. Parse Prolog Predicate         │
│     - Extract clauses               │
│     - Analyze structure             │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  2. Classify Recursion Pattern     │
│     - Non-recursive                 │
│     - Tail recursion                │
│     - Linear recursion              │
│     - Tree recursion                │
│     - Mutual recursion              │
│     - Basic recursion (BFS)         │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  3. Analyze Constraints            │
│     - Unique (single result)        │
│     - Unordered (any order)         │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  4. Select Optimization Strategy   │
│     - BFS for transitive closure    │
│     - Loop for tail recursion       │
│     - Memoization for linear        │
│     - Structural for tree           │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  5. Generate Bash Code             │
│     - Select template               │
│     - Fill placeholders             │
│     - Add error handling            │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  6. Output Bash Script             │
│     - Function definitions          │
│     - Helper functions              │
│     - Memoization tables            │
└─────────────────────────────────────┘
```

### Template System

UnifyWeaver uses a mustache-style template engine:

**Template:**
```bash
{{predicate_name}}_all() {
    local arg1="$1"
    local arg2="$2"

    {{#base_clauses}}
    # Base case
    if [[ "{{condition}}" ]]; then
        echo "{{result}}"
    fi
    {{/base_clauses}}

    {{#recursive_logic}}
    # Recursive logic here
    {{/recursive_logic}}
}
```

**Placeholders:**
- `{{predicate_name}}` - Name of the predicate
- `{{#section}}...{{/section}}` - Conditional sections
- `{{variable}}` - Variable substitution

---

## Testing Guide

### Test Environment

UnifyWeaver includes comprehensive test infrastructure:

```bash
cd scripts/testing
./init_testing.sh  # or Init-TestEnvironment.ps1 on Windows
```

### Test Suites

**Core Tests:**
```prolog
?- test_template_system.      % Template rendering
?- test_stream_compiler.      % Non-recursive compilation
?- test_recursive_compiler.   % Recursive compilation
?- test_constraint_system.    % Constraint detection
```

**Advanced Tests:**
```prolog
?- test_advanced.             % All advanced patterns
?- test_tail_recursion.       % Tail → loop
?- test_linear_recursion.     % Linear → memo
?- test_tree_recursion.       % Tree → structural
?- test_mutual_recursion.     % Mutual → shared memo
```

**Data Source Tests:**
```prolog
?- test_csv_source.           % CSV plugin
?- test_json_source.          % JSON plugin
?- test_http_source.          % HTTP plugin
?- test_python_source.        % Python plugin
```

**Firewall Tests:**
```prolog
?- test_firewall.             % Firewall basics
?- test_network_firewall.     % Network access control
```

### Writing New Tests

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for detailed instructions on adding tests.

---

## Troubleshooting

### Common Issues

**1. SIGPIPE Errors with `head`**

**Symptom:**
```bash
bash: line 10: echo: write error: Broken pipe
```

**Cause:** When piping to `head`, the pipe closes early, causing SIGPIPE.

**Solution:**
```bash
# Option 1: Redirect stderr
ancestor_all alice | head -5 2>/dev/null

# Option 2: Write to temp file first
ancestor_all alice > /tmp/results.txt
head -5 /tmp/results.txt
```

**2. Associative Array Errors**

**Symptom:**
```bash
declare: -A: invalid option
```

**Cause:** Bash version < 4.0

**Solution:** Upgrade to Bash 4.0+:
```bash
bash --version  # Check version
# Upgrade via package manager or WSL
```

**3. Process ID Collisions**

**Symptom:** Temp files from different processes conflict

**Cause:** Older templates used `$` instead of `$$`

**Solution:** Regenerate scripts with latest templates, or manually fix:
```bash
# Old (wrong):
temp_file="/tmp/ancestor_$"

# New (correct):
temp_file="/tmp/ancestor_$$"
```

**4. Module Import Conflicts**

**Symptom:**
```
ERROR: source/3 is already imported from module X
```

**Cause:** Multiple modules define `source/3`

**Solution:** Use module-qualified calls:
```prolog
% Instead of:
source(csv, users, [...]).

% Use:
csv_source:source(csv, users, [...]).
```

**5. Firewall Blocking Operations**

**Symptom:**
```
[Firewall] Operation denied: network_access(https://example.com)
```

**Cause:** Firewall policy blocks the operation

**Solution:** Update firewall policy:
```prolog
% Add to allowed domains
assertz(allowed_domain('example.com')).

% Or load permissive policy
load_firewall_policy(permissive).
```

### Getting Help

1. Check [TESTING.md](TESTING.md) for test examples
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. See [ADVANCED_RECURSION.md](ADVANCED_RECURSION.md) for recursion patterns
4. Open an issue on [GitHub](https://github.com/s243a/UnifyWeaver/issues)

---

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
* MIT license ([LICENSE-MIT](../LICENSE-MIT))

at your option.

---

## Acknowledgments

**Contributors:**
- John William Creighton (@s243a) - Core development
- Gemini (via gemini-cli) - Constraint awareness features
- Claude (via Claude Code) - Advanced recursion, testing, documentation

---

**Last Updated:** 2025-12-17
**Version:** 0.9.0

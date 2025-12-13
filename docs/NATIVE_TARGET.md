# Native Targets

UnifyWeaver supports native compilation targets for high-performance JSONL pipeline processing.

## Status

| Target | Module | Status |
|--------|--------|--------|
| C | `c_target.pl` | ✅ Initial (pipeline + generator modes) |
| C++ | `cpp_target.pl` | ✅ Initial (pipeline + generator modes) |
| Rust | `rust_target.pl` | ✅ Mature |
| Go | `go_target.pl` | ✅ Mature |

## Quick Start

### C Target
```prolog
?- use_module('src/unifyweaver/targets/c_target').
?- compile_predicate_to_c(filter/2, [pipeline_input(true)], Code).
```

### C++ Target
```prolog
?- use_module('src/unifyweaver/targets/cpp_target').
?- compile_predicate_to_cpp(filter/2, [pipeline_input(true)], Code).
```

## Options

| Option | Values | Description |
|--------|--------|-------------|
| `pipeline_input(Bool)` | true/false | Enable streaming JSONL mode |
| `generator_mode(Bool)` | true/false | Enable generator/iterator mode |
| `program_name(Name)` | atom | Executable name for Makefile |
| `project_name(Name)` | atom | Project name for CMake |

## JSON Libraries

| Target | Library | Notes |
|--------|---------|-------|
| C | [cJSON](https://github.com/DaveGamble/cJSON) | MIT, header-only compatible |
| C++ | [nlohmann/json](https://github.com/nlohmann/json) | MIT, header-only |

## Build System Generation

### C Target
```prolog
% Generate Makefile
?- generate_makefile([program_name(my_pipeline)], Makefile).

% Generate CMakeLists.txt
?- generate_cmake([project_name('MyPipeline')], CMake).
```

### C++ Target
```prolog
% Generate CMakeLists.txt with FetchContent
?- generate_cmake_cpp([project_name('MyCppPipeline')], CMake).
```

CMake FetchContent automatically downloads nlohmann/json.

## Recursion Patterns

Both targets support recursion optimization:

| Pattern | C Implementation | C++ Implementation |
|---------|-----------------|-------------------|
| Tail Recursion | `while` loop | `for` loop with `constexpr` limit |
| General Recursion | Explicit stack array | `std::vector<json>` stack |

## Bindings

### C Bindings (41 total)
- **Stdlib**: malloc, free, calloc, realloc, atoi, atof, exit
- **I/O**: printf, fprintf, fgets, fopen, fread, fwrite
- **Strings**: strlen, strcmp, strcpy, strdup, strtok, strstr
- **cJSON**: cJSON_Parse, cJSON_Delete, cJSON_Print, cJSON_GetObjectItem

### C++ Bindings (45 total)
- **STL**: vector, map, string, optional operations
- **iostream**: cout, cin, getline, fstream
- **Algorithms**: std::find, std::sort, std::transform, std::accumulate
- **nlohmann/json**: parse, dump, at, value, contains

## Example Generated Code

### C Pipeline
```c
#include <stdio.h>
#include "cJSON.h"

cJSON* process(cJSON* record) {
    // Process and return, or NULL to filter
    return record;
}

void run_pipeline(void) {
    char line[65536];
    while (fgets(line, sizeof(line), stdin)) {
        cJSON* record = cJSON_Parse(line);
        cJSON* result = process(record);
        if (result) {
            printf("%s\n", cJSON_PrintUnformatted(result));
        }
        cJSON_Delete(record);
    }
}
```

### C++ Pipeline
```cpp
#include <iostream>
#include <optional>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::optional<json> process(const json& record) {
    // Return result or std::nullopt to filter
    return record;
}

void runPipeline() {
    std::string line;
    while (std::getline(std::cin, line)) {
        json record = json::parse(line);
        if (auto result = process(record)) {
            std::cout << result->dump() << std::endl;
        }
    }
}
```

## Tested Compilers

| Compiler | Version | Status |
|----------|---------|--------|
| gcc | 9.4.0 | ✅ Tested |
| g++ | 9.4.0 | ✅ Tested |
| clang | - | Should work |

# UnifyWeaver v0.0.2 Data Sources Implementation Plan

**Target Release:** v0.0.2  
**Planning Date:** October 14, 2025  
**Status:** Implementation Ready  
**Estimated Duration:** 2-3 weeks  

---

## Overview

This document outlines the implementation plan for UnifyWeaver v0.0.2's comprehensive data source system. Building on the existing `awk_source.pl` pattern, we will implement 4 additional data sources plus enhanced firewall capabilities.

### Features Being Added

1. **CSV/TSV Source** - Structured data processing
2. **Python Embedded Source** - Complex transformations + SQLite support
3. **HTTP Source** - Network data fetching via curl/wget
4. **JSON Source** - JSON processing via jq
5. **Enhanced Firewall** - Multi-service security model

---

## Architecture Overview

### Current Pattern (from awk_source.pl)

Each data source plugin follows this interface:

```prolog
:- module(source_name, [
    compile_source/4,          % +Pred/Arity, +Config, +Options, -BashCode
    validate_config/1,         % +Config
    source_info/1              % -Info
]).

% Self-registration
:- initialization(register_source_type(type_name, module_name), now).

% Template-based bash generation
generate_bash(PredStr, Arity, Config, BashCode) :-
    render_named_template(template_name, Variables, [], BashCode).
```

### Plugin Integration Flow

```
User Code:
:- source(csv, users, [csv_file('data.csv'), has_header(true)]).

    ↓

Registration System:
register_source_type(csv, csv_source)

    ↓

Compilation Request:
compile_source(csv, users, [...], BashCode)

    ↓

Template Rendering:
render_named_template(csv_with_header, [pred='users', file='data.csv'], BashCode)

    ↓

Generated Bash Function:
users() { awk -F, 'NR>1 {print $1":"$2":"$3}' data.csv; }
```

---

## Implementation Details

## 1. CSV/TSV Source 📊

### **File:** `src/unifyweaver/sources/csv_source.pl`

### Architecture

**Configuration Options:**
```prolog
csv_config([
    csv_file('data.csv'),              % Input file path
    delimiter(','),                    % Field separator: ',' '\t' or custom
    has_header(true),                  % Auto-generate predicate from headers
    columns([name, age, city]),        % Manual column specification
    skip_lines(0),                     % Skip N lines (comments/metadata)
    quote_char('"'),                   % Quote character for embedded delimiters
    quote_handling(strip)              % strip, preserve, or escape
]).
```

**Template Strategy:**

For header auto-detection:
```bash
{{pred}}() {
    # Auto-generated from CSV headers: name,age,city
    awk -F'{{delimiter}}' '
    NR > {{skip_lines}} + 1 {
        gsub(/{{quote_char}}/, "", $0)  # Handle quotes
        print $1":"$2":"$3             # Based on column count
    }' {{file}}
}
```

---

## 2. Python Embedded Source 🐍

### **File:** `src/unifyweaver/sources/python_source.pl`

### Architecture

**Configuration Options:**
```prolog
python_config([
    python_inline('import sys; print("hello")'),   % Inline Python code
    python_file('script.py'),                      % External Python file
    sqlite_query('SELECT * FROM users'),           % SQLite shorthand
    database('app.db'),                            % Database file
    input_mode(stdin),                             % stdin, args, or none
    output_format(tsv),                            % tsv, json, or custom
    timeout(30),                                   % Execution timeout
    python_interpreter('python3')                  % python3, python, custom
]).
```

**Template Strategy (Heredoc Pattern):**

```bash
{{pred}}() {
    local timeout_val="{{timeout}}"
    
    timeout "$timeout_val" {{python_interpreter}} /dev/fd/3 "$@" 3<<'PYTHON'
{{python_code}}
PYTHON
}
```

**Specialized SQLite Template:**
```bash
{{pred}}() {
    {{python_interpreter}} /dev/fd/3 "$@" 3<<'PYTHON'
import sqlite3
import sys

try:
    conn = sqlite3.connect('{{database}}')
    cursor = conn.execute('''{{query}}''')
    
    for row in cursor:
        print(':'.join(str(x) if x is not None else '' for x in row))
        
    conn.close()
except sqlite3.Error as e:
    print(f"SQLite error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON
}
```

---

## 3. HTTP Source 🌐

### **File:** `src/unifyweaver/sources/http_source.pl`

**Configuration Options:**
```prolog
http_config([
    url('https://api.example.com/users'),          % Target URL
    method(get),                                   % get, post, put, delete
    headers(['Authorization: Bearer token']),       % HTTP headers
    cache_file('/tmp/api_cache.json'),             % Response caching
    cache_duration(300),                           % Cache TTL (seconds)
    timeout(30),                                   % Request timeout
    tool(curl)                                     % curl or wget
]).
```

**Template Strategy:**
```bash
{{pred}}() {
    local cache_file="{{cache_file}}"
    local cache_duration="{{cache_duration}}"
    
    # Check cache validity
    if [[ -f "$cache_file" ]]; then
        local cache_age=$(( $(date +%s) - $(stat -c%Y "$cache_file") ))
        if [[ $cache_age -lt $cache_duration ]]; then
            cat "$cache_file"
            return
        fi
    fi
    
    # Make HTTP request
    curl -s \
         -X {{method}} \
         {{#headers}}-H '{{.}}' {{/headers}} \
         --max-time {{timeout}} \
         "{{url}}" | tee "$cache_file"
}
```

---

## 4. JSON Source 📄

### **File:** `src/unifyweaver/sources/json_source.pl`

**Configuration Options:**
```prolog
json_config([
    json_file('data.json'),                    % Input JSON file
    json_stdin(true),                          % Read from stdin
    jq_filter('.users[] | {id, name}'),        % jq filter expression
    output_format(tsv),                        % tsv, json, raw
    raw_output(true)                           % -r flag for jq
]).
```

**Template Strategy:**
```bash
{{pred}}() {
    jq -r '{{jq_filter}}' {{json_file}}
}
```

---

## 5. Enhanced Firewall System 🔒

### **File:** `src/unifyweaver/core/firewall.pl` (update)

**Enhanced Firewall:**
```prolog
:- firewall([
    target(bash),
    services([awk, python3, curl, wget, jq]),
    python_modules([sys, json, sqlite3, csv, re]),
    network_access(allowed),
    output_dirs(['output/*', 'data/*'])
]).
```

**Service Validation:**
```prolog
validate_service(Service, Firewall) :-
    member(services(AllowedServices), Firewall),
    (   member(Service, AllowedServices)
    ->  true
    ;   format(user_error, 'Firewall blocks service: ~w~n', [Service]),
        fail
    ).
```

---

## Implementation Timeline

### **Week 1: Foundation**

**Days 1-2: CSV Source**
- [ ] Implement `csv_source.pl` module
- [ ] Header auto-detection logic
- [ ] Quote handling for embedded delimiters
- [ ] Unit tests for basic functionality
- [ ] TSV support verification

**Days 3-4: Python Source** 
- [ ] Implement `python_source.pl` module
- [ ] Heredoc template with /dev/fd/3 pattern
- [ ] SQLite integration templates
- [ ] Input/output mode handling
- [ ] Error handling and timeouts

**Day 5: Firewall Enhancement**
- [ ] Extend `firewall.pl` with multi-service support
- [ ] Service validation predicates
- [ ] Network access validation
- [ ] Python module restriction system

### **Week 2: Network & JSON**

**Days 1-2: HTTP Source**
- [ ] Implement `http_source.pl` module
- [ ] curl template support
- [ ] Caching mechanism implementation
- [ ] Header and POST data handling
- [ ] Timeout and error handling

**Days 3-4: JSON Source**
- [ ] Implement `json_source.pl` module
- [ ] jq integration templates
- [ ] Filter validation
- [ ] File and stdin input modes
- [ ] Integration with HTTP source

**Day 5: Integration Testing**
- [ ] Cross-source pipeline tests
- [ ] Performance testing
- [ ] Error condition testing
- [ ] Firewall integration validation

### **Week 3: Polish & Documentation**

**Days 1-2: Documentation**
- [ ] Create `docs/DATA_SOURCES.md` overview
- [ ] Individual source documentation files
- [ ] Update `README.md` with new capabilities
- [ ] Create example use cases

**Days 3-4: Examples & Tutorials**
- [ ] Real-world example implementations
- [ ] Tutorial walkthroughs
- [ ] Performance comparison documentation
- [ ] Best practices guide

**Day 5: Release Preparation**
- [ ] Final integration testing
- [ ] Version number updates
- [ ] CHANGELOG.md updates
- [ ] Release notes preparation

---

## Example Use Cases

### Use Case 1: API → Database ETL Pipeline

```prolog
% Fetch user data from GitHub API
:- source(http, github_users, [
    url('https://api.github.com/users'),
    headers(['Authorization: token ghp_xxxx']),
    cache_file('/tmp/github_users.json'),
    cache_duration(3600)
]).

% Parse JSON response to extract relevant fields
:- source(json, parse_users, [
    jq_filter('.[] | [.login, .id, .html_url, .type] | @tsv'),
    output_format(tsv)
]).

% Store in SQLite database
:- source(python, store_users, [
    sqlite_query('''
        INSERT OR REPLACE INTO users (login, github_id, url, type) 
        VALUES (?, ?, ?, ?)
    '''),
    database('github_users.db')
]).

% Usage: github_users | parse_users | store_users
```

### Use Case 2: CSV Data Analysis

```prolog
% Read sales data with auto-detected headers
:- source(csv, sales_data, [
    csv_file('monthly_sales.csv'),
    has_header(true)
]).

% Compute monthly statistics
:- source(python, sales_stats, [
    python_inline('
import sys
from collections import defaultdict

totals = defaultdict(float)
counts = defaultdict(int)

for line in sys.stdin:
    month, amount, category = line.strip().split(":")
    totals[month] += float(amount)
    counts[month] += 1

for month in sorted(totals.keys()):
    avg = totals[month] / counts[month] if counts[month] > 0 else 0
    print(f"{month}:{totals[month]:.2f}:{avg:.2f}")
')
]).

% Usage: sales_data | sales_stats
```

---

## Testing Strategy

### Unit Tests Structure

**File:** `tests/core/test_data_sources.pl`

```prolog
:- module(test_data_sources, [test_data_sources/0]).

test_data_sources :-
    test_csv_source,
    test_python_source,
    test_http_source,
    test_json_source,
    test_enhanced_firewall.
```

**CSV Source Tests:**
```prolog
test_csv_with_headers :-
    % Create test CSV file
    write_test_csv('test_headers.csv', 'name,age,city\nalice,25,nyc\nbob,30,sf\n'),
    
    % Compile source
    compile_source(csv, users, [
        csv_file('test_headers.csv'), 
        has_header(true)
    ], Code),
    
    % Verify generated code
    sub_string(Code, _, _, _, 'users()'),
    sub_string(Code, _, _, _, 'awk'),
    
    % Clean up
    delete_file('test_headers.csv').
```

**Python Source Tests:**
```prolog
test_python_inline :-
    compile_source(python, hello, [
        python_inline('print("hello:world")')
    ], Code),
    
    % Check heredoc pattern
    sub_string(Code, _, _, _, '/dev/fd/3'),
    sub_string(Code, _, _, _, 'PYTHON'),
    sub_string(Code, _, _, _, 'print("hello:world")').
```

---

## File Structure After Implementation

```
src/unifyweaver/sources/
├── awk_source.pl           # Existing
├── csv_source.pl           # NEW
├── python_source.pl        # NEW  
├── http_source.pl          # NEW
└── json_source.pl          # NEW

src/unifyweaver/core/
├── firewall.pl             # Enhanced
└── ...

tests/core/
├── test_data_sources.pl    # NEW
├── test_csv_source.pl      # NEW
├── test_python_source.pl   # NEW
├── test_http_source.pl     # NEW
├── test_json_source.pl     # NEW
└── test_firewall_enhanced.pl # NEW

docs/
├── DATA_SOURCES.md         # NEW - Overview
├── CSV_SOURCE.md           # NEW - CSV/TSV guide
├── PYTHON_SOURCE.md        # NEW - Python + SQLite
├── HTTP_SOURCE.md          # NEW - Network data
└── JSON_SOURCE.md          # NEW - jq integration

examples/
├── api_to_database/        # NEW - Complete ETL example
├── csv_analysis/           # NEW - Data analysis example
└── config_management/      # NEW - Configuration example
```

---

## Success Criteria

### Functional Requirements ✅

- [ ] All 5 data sources compile valid bash code
- [ ] Generated functions produce correct output
- [ ] Firewall properly validates service access
- [ ] Error handling works for edge cases
- [ ] Performance meets baseline requirements

### Quality Requirements ✅

- [ ] 90%+ test coverage for new code
- [ ] All existing tests continue passing
- [ ] Documentation covers all new features
- [ ] Examples demonstrate real-world usage
- [ ] Code follows established patterns

---

## Next Steps

1. **Create feature branch:** `new-data-sources` ✅
2. **Begin implementation** following the timeline
3. **Regular progress reviews** and testing
4. **Documentation and examples** creation
5. **Release preparation** and deployment

---

*This plan was created on October 14, 2025, and serves as the definitive implementation guide for UnifyWeaver v0.0.2 data sources functionality.*

<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Chapter 1: Perl Service Infrastructure

**Status:** ✅ Implemented (v0.1, merged 2025-11-05)
**Module:** `src/unifyweaver/core/perl_service.pl`
**Tests:** `tests/core/test_perl_service.pl` (1/1 passing)

---

## Introduction

The Perl service infrastructure enables UnifyWeaver to generate inline Perl code execution within bash scripts. This provides a Python-free alternative for data processing tasks, making UnifyWeaver deployable in minimal environments where installing Python packages is difficult or impossible.

### Primary Use Case: XML Processing

The first application of Perl services is the XML splitter for the `xmllint` engine, which previously required Python/lxml for record splitting. With Perl services, xmllint can now function without Python dependencies.

### Design Philosophy

The Perl service module provides **bash code generation** rather than direct Perl execution. It:
- Generates bash snippets that invoke Perl via heredocs
- Handles shell quoting automatically
- Avoids label collisions in nested heredocs
- Integrates with UnifyWeaver's template system

---

## Architecture

### Module Interface

```prolog
:- module(perl_service, [
    check_perl_available/0,
    generate_inline_perl_call/4
]).
```

**Key Predicates:**

1. `check_perl_available/0` - Verify Perl interpreter is installed
2. `generate_inline_perl_call(+PerlCode, +Args, +InputVar, -BashCode)` - Generate bash wrapper

### Code Generation Flow

```
User Prolog Code
    ↓
perl_service.pl
    ↓
DCG-based Code Generation
    ↓
Bash Script with Heredoc
    ↓
Perl Execution at Runtime
```

---

## Example: Basic Usage

### Generate a Simple Perl Call

```prolog
?- generate_inline_perl_call(
    'print "Hello from Perl\n"',
    [],
    stdin,
    BashCode
).

BashCode = 'perl /dev/fd/3 3<<\'PERL\'
print "Hello from Perl\n"
PERL
'.
```

### Generated Bash Output

```bash
perl /dev/fd/3 3<<'PERL'
print "Hello from Perl\n"
PERL
```

---

## Key Features

### 1. Heredoc Label Collision Avoidance

**Problem:** Nested heredocs or Perl code containing the label "PERL" would break.

**Solution:** `choose_heredoc_label/2` automatically finds a unique label.

```prolog
% If 'PERL' appears in code, use 'PERL_END'
% If 'PERL_END' appears, use 'PERL_END_END', etc.
choose_heredoc_label(PerlCode, Label).
```

**Example:**

```prolog
?- choose_heredoc_label('This contains PERL keyword', Label).
Label = 'PERL_END'.

?- choose_heredoc_label('PERL and PERL_END both here', Label).
Label = 'PERL_END_END'.
```

### 2. Safe Shell Quoting

**Problem:** Arguments containing special characters (`'`, `"`, `$`, backticks) break bash.

**Solution:** `shell_quote_arg/2` wraps arguments in single quotes and escapes embedded quotes.

```prolog
shell_quote_arg('user\'s data', Quoted).
% Quoted = '\'user'\''s data\''
```

**Generated Bash:**
```bash
perl script.pl 'user'\''s data'
```

### 3. DCG-Based Code Generation

The module uses Definite Clause Grammars (DCGs) for clean, composable code generation:

```prolog
perl_call_codes(ArgSegment, LabelCodes, InputRedirect, Body) -->
    "perl /dev/fd/3",
    arg_segment(ArgSegment),
    " 3<<'",
    LabelCodes,
    "'",
    InputRedirect,
    "\n",
    Body,
    LabelCodes,
    "\n".
```

This generates well-formed bash regardless of input complexity.

### 4. Input Redirection Modes

**Mode 1: stdin (caller provides input)**
```prolog
generate_inline_perl_call(Code, [], stdin, Bash).
% Caller must pipe data or redirect stdin
```

Generated bash expects input from caller:
```bash
echo "data" | perl /dev/fd/3 3<<'PERL'
...
PERL
```

**Mode 2: Variable input (here-string)**
```prolog
generate_inline_perl_call(Code, [], my_var, Bash).
```

Generated bash reads from variable:
```bash
perl /dev/fd/3 3<<'PERL' <<< "$my_var"
...
PERL
```

---

## Real-World Example: XML Splitter

### The Problem

The xmllint engine needed to:
1. Run `xmllint --xpath` to extract XML fragments
2. Split null-separated fragments into individual records
3. Repair namespace declarations in each fragment

Previously required Python/lxml. Now uses Perl.

### The Solution

```prolog
% In xml_source.pl
generate_perl_splitter_code(Tags, NamespaceFix, BashCode) :-
    % Build Perl script
    format(string(PerlScript), '...split and repair logic...', [...]),

    % Generate bash wrapper
    generate_inline_perl_call(
        PerlScript,
        [repair_flag, ns_map | Tags],
        stdin,
        BashCode
    ).
```

### Generated Bash

```bash
xmllint --xpath '//Record' file.xml | \
perl /dev/fd/3 'true' 'pt=http://...' 'Record' 3<<'PERL'
#!/usr/bin/env perl
use strict;
use warnings;

my $repair = shift @ARGV;
my $ns_map_str = shift @ARGV;
my @tags = @ARGV;

# ... (splitting and namespace repair logic)
PERL
```

---

## Testing

### Unit Tests

**Location:** `tests/core/test_perl_service.pl`

```prolog
:- begin_tests(perl_service).

test(inline_perl_call, [true]) :-
    % Test basic call generation
    generate_inline_perl_call(
        'print "test\n"',
        ['arg1', 'arg2'],
        stdin,
        BashCode
    ),
    % Verify output structure
    atom_string(BashCode, Str),
    sub_string(Str, _, _, _, "perl /dev/fd/3"),
    sub_string(Str, _, _, _, "arg1"),
    sub_string(Str, _, _, _, "print \"test\\n\"").

:- end_tests(perl_service).
```

**Run Tests:**
```bash
swipl -s tests/core/test_perl_service.pl -g run_tests -t halt
# [1/1] perl_service:inline_perl_call ............... passed (0.037 sec)
```

---

## Integration with Other Systems

### Firewall Awareness

Perl service integrates with `firewall_v2` and `tool_detection`:

```prolog
% Check if Perl is available
check_perl_available :-
    perl_candidates(Candidates),
    member(Exec-Args, Candidates),
    perl_check(Exec, Args),
    !.
```

If Perl is denied by firewall or unavailable, compilation falls back to other strategies.

### Preference System

Users can prefer Perl over Python for XML processing:

```prolog
:- prefer([xmllint_splitter(perl)]).

xml_source(my_data/1, [
    file('data.xml'),
    xpath('//Record'),
    engine(xmllint)  % Will use Perl splitter
]).
```

---

## Limitations & Future Work

### Current Limitations

1. **No Perl library dependencies** - Only uses core Perl modules (strict, warnings)
2. **No CPAN module support** - Would require installation in target environment
3. **Simple I/O model** - stdin/stdout only, no file handles
4. **No error propagation** - Perl errors appear in bash stderr but don't halt compilation

### Potential Extensions

1. **AWK alternative** - Generate AWK code instead of Perl for even wider compatibility
2. **Multi-stage processing** - Chain multiple Perl calls in a pipeline
3. **Template integration** - Allow Perl scripts in template system
4. **Library detection** - Check for optional CPAN modules and adapt functionality

---

## Performance Considerations

### Perl vs Python

**Advantages of Perl:**
- Nearly universal availability (pre-installed on most Unix systems)
- Fast startup time
- Good regex performance
- No package installation required

**Disadvantages of Perl:**
- Slower than Python for numeric computation
- Less readable for complex data structures
- Fewer modern libraries

### When to Use Each

| Use Case | Recommended | Why |
|----------|-------------|-----|
| XML splitting | Either | Performance similar |
| Text processing | Perl | Faster startup, better regex |
| JSON parsing | Python | Better libraries |
| Large file processing | Python | lxml streaming is more efficient |
| Minimal environments | Perl | No installation required |

---

## See Also

- **Chapter 14 (Book 1):** XML Source Plugin - Uses Perl service for splitting
- **Book 1, Chapter 8:** Template System - Could integrate Perl generation
- **Appendix A:** SIGPIPE and Streaming Safety - Applies to Perl pipelines
- **Module:** `src/unifyweaver/core/tool_detection.pl` - Perl detection

---

## Summary

The Perl service infrastructure provides:
- ✅ Python-free alternative for data processing
- ✅ Clean bash code generation via DCGs
- ✅ Safe shell quoting and heredoc handling
- ✅ Integration with firewall and preferences
- ✅ Minimal dependencies (core Perl only)

This enables UnifyWeaver deployment in restricted environments while maintaining full XML processing capabilities.

---

**Next Chapter:** [02: XML Perl Splitter](02_xml_perl_splitter.md)

---

**Authors:** John William Creighton (@s243a), Claude Code (Sonnet 4.5)
**Last Updated:** 2025-11-05

# Perl Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Perl code generation target testing

## Overview

This test plan covers the Perl target for UnifyWeaver, which generates Perl scripts for text processing, system administration, and legacy integration.

## Prerequisites

### System Requirements

- Perl 5.26+ (5.32+ recommended)
- CPAN modules: JSON, Text::CSV (optional)
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Perl installation
perl --version

# Check core modules
perl -MJSON -e 'print "JSON OK\n"'
perl -MText::CSV -e 'print "CSV OK\n"' 2>/dev/null || echo "Text::CSV not installed"

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_perl_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `scalar_variables` | $var syntax | Correct sigils |
| `array_variables` | @array syntax | Array operations |
| `hash_variables` | %hash syntax | Hash operations |
| `subroutine_generation` | sub definitions | Correct sub syntax |
| `regex_patterns` | m// and s/// | Perl regex syntax |
| `file_handling` | open/close | Filehandle operations |

#### 1.2 Data Structure Tests

```bash
swipl -g "use_module('tests/core/test_perl_data_structures'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `array_of_hashes` | Complex structures | Correct dereferencing |
| `hash_of_arrays` | Nested structures | Arrow notation |
| `references` | \$, \@, \% | Reference creation |

### 2. Integration Tests

#### 2.1 Script Execution

```bash
./tests/integration/test_perl_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `basic_execution` | Run generated script | Exit code 0 |
| `stdin_processing` | Read from STDIN | Correct input handling |
| `regex_matching` | Pattern matching | Matches found |
| `json_output` | JSON emission | Valid JSON |

#### 2.2 One-Liner Tests

```bash
./tests/integration/test_perl_oneliners.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `perl_e` | -e flag execution | Inline code works |
| `perl_n` | -n flag (implicit loop) | Line processing |
| `perl_p` | -p flag (print loop) | Auto-print works |
| `perl_a` | -a flag (autosplit) | @F populated |

### 3. Generated Code Structure

```perl
#!/usr/bin/env perl
use strict;
use warnings;
use JSON;

my %facts;
my %delta;

# Initialize facts
sub init_facts {
    $facts{"parent"}{"john"}{"mary"} = 1;
    # ...
}

# Apply rules
sub apply_rules {
    my $changed = 1;
    while ($changed) {
        $changed = 0;
        # Rule application
    }
}

# Main
init_facts();
apply_rules();
print encode_json(\%facts);
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/perl_target'),
    compile_to_perl(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_perl_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PERL5LIB` | Module search path | (system) |
| `SKIP_PERL_EXECUTION` | Skip runtime tests | `0` |
| `KEEP_PERL_ARTIFACTS` | Preserve generated code | `0` |

## Known Issues

1. **Unicode handling**: Requires `use utf8` and binmode
2. **CPAN dependencies**: Some features need external modules
3. **Perl 7 compatibility**: Future syntax changes pending

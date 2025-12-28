# Perl Target

The Perl target (`target(perl)`) generates Perl subroutines from Prolog predicates using continuation-passing style (CPS). It provides streaming evaluation with callback-based output, suitable for Unix pipeline integration and data processing.

## Overview

Perl programs use callback functions to stream results, avoiding the need to materialize complete result sets in memory. This makes them efficient for processing large data sets in a pipeline context.

```prolog
% Compile to Perl
?- compile_predicate_to_perl(my_predicate/2, [], PerlCode).

% With JSON output wrapper
?- compile_predicate_to_perl(my_predicate/2, [json_output], PerlCode).
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Facts | ✅ | Direct fact compilation to arrays |
| Single Rules | ✅ | Body-to-code translation with CPS |
| Multiple Rules (OR) | ✅ | Sequential clause evaluation |
| Semi-naive Recursion | ✅ | Datalog-style fixpoint with deduplication |
| Tail Recursion | ✅ | Accumulator patterns optimized to loops |
| Linear Recursion | ✅ | Memoized recursive calls (fibonacci-style) |
| Joins | ✅ | Inner joins with proper variable binding |
| Aggregations | ✅ | count, sum, min, max, avg |
| JSON Output | ✅ | JSON array wrapper with `encode_json` |
| Pipeline Mode | ✅ | stdin/stdout JSON streaming |

## Compilation Options

| Option | Description |
|--------|-------------|
| `json_output` | Generate `_json` wrapper function with JSON output |
| `json_input` | Read input facts from JSON (planned) |
| `pipeline` | Generate `run_pipeline` for shell integration |

## Continuation-Passing Style

All predicates generate subroutines that take a callback as the first argument:

```perl
sub parent {
    my $callback = shift;
    my @facts = (
        ["alice", "bob"],
        ["bob", "charlie"]
    );
    foreach my $fact (@facts) {
        $callback->(@$fact);
    }
}

# Usage: stream all parent relationships
parent(sub { my ($x, $y) = @_; print "$x -> $y\n"; });
```

## Join Handling

When rules contain multiple goals sharing variables, the compiler generates proper join conditions:

```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

Generates:

```perl
sub grandparent {
    my $callback = shift;
    parent(sub {
        my ($p0, $p1) = @_;
        parent(sub {
            my ($p2, $p3) = @_;
            return unless $p2 eq $p1;  # Join condition
            $callback->($p0, $p3);
        });
    });
}
```

Key implementation details:
- **Unique parameter names** (`$p0`, `$p1`, etc.) avoid variable shadowing in nested callbacks
- **Join conditions** (`return unless`) enforce variable equality between goals
- **Immediate projection** returns only the head variables to the callback

## Recursion Patterns

### Semi-naive Recursion (Datalog-style)

For transitive closure patterns, the compiler generates semi-naive iteration:

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

Generates:

```perl
sub ancestor {
    my $callback = shift;
    my @delta;
    my %seen;

    # Base cases - seed the worklist
    parent(sub {
        my ($p0, $p1) = @_;
        my $key = join('\0', $p0, $p1);
        unless ($seen{$key}++) { push @delta, [$p0, $p1]; }
    });

    # Semi-naive iteration
    while (@delta) {
        my $item = shift @delta;
        $callback->(@$item);

        parent(sub {
            my ($p2, $p3) = @_;
            return unless $p3 eq $item->[0];
            my $key = join('\0', $p2, $item->[1]);
            unless ($seen{$key}++) { push @delta, [$p2, $item->[1]]; }
        });
    }
}
```

### Tail Recursion Optimization

Accumulator-style predicates are optimized to efficient loops:

```prolog
factorial(0, Acc, Acc).
factorial(N, Acc, Result) :-
    N > 0,
    N1 is N - 1,
    Acc1 is Acc * N,
    factorial(N1, Acc1, Result).
```

Generates:

```perl
sub factorial {
    my $callback = shift;
    my ($arg1, $arg2, $arg3) = @_;

    # Tail recursion optimized to loop
    while (1) {
        if ($arg1 == 0) {
            $callback->($arg2);
            return;
        }
        my $tmp0 = ($arg1 - 1);
        my $tmp1 = ($arg2 * $arg1);
        $arg1 = $tmp0;
        $arg2 = $tmp1;
    }
}
```

### Linear Recursion with Memoization

Fibonacci-style predicates with multiple recursive calls get automatic memoization:

```prolog
fib(0, 0).
fib(1, 1).
fib(N, Result) :-
    N > 1,
    N1 is N - 1, N2 is N - 2,
    fib(N1, R1), fib(N2, R2),
    Result is R1 + R2.
```

Generates:

```perl
{
    my %memo;
    sub fib {
        my $callback = shift;
        my ($arg1) = @_;
        my $key = join('\0', $arg1);
        if (exists $memo{$key}) {
            return $callback->($memo{$key});
        }
        if ($arg1 == 0) {
            $memo{$key} = 0;
            return $callback->(0);
        }
        if ($arg1 == 1) {
            $memo{$key} = 1;
            return $callback->(1);
        }

        my $tmp0 = ($arg1 - 1);
        my $tmp1 = ($arg1 - 2);
        fib(sub {
            my ($r0) = @_;
            fib(sub {
                my ($r1) = @_;
                my $result = ($r0 + $r1);
                $memo{$key} = $result;
                $callback->($result);
            }, $tmp1);
        }, $tmp0);
    }
}
```

## Aggregations

The Perl target supports `aggregate_all/3` with common aggregation templates:

```prolog
total_price(Total) :-
    aggregate_all(sum(Price), item(_, Price), Total).
```

Generates:

```perl
sub total_price {
    my $callback = shift;
    my $agg_result = 0;

    item(sub {
        my ($g1, $g2) = @_;
        my $tmpl_val = $g2;
        $agg_result += $tmpl_val;
    });

    $callback->($agg_result);
}
```

### Supported Aggregation Templates

| Template | Init | Update | Description |
|----------|------|--------|-------------|
| `count` | `0` | `+= 1` | Count matching tuples |
| `sum(X)` | `0` | `+= X` | Sum of values |
| `min(X)` | `undef` | `= X if !defined or X <` | Minimum value |
| `max(X)` | `undef` | `= X if !defined or X >` | Maximum value |
| `avg(X)` | `[0, 0]` | `sum += X; count++` | Average (sum/count) |

## JSON Output Mode

The `json_output` option generates an executable wrapper:

```prolog
?- compile_predicate_to_perl(parent/2, [json_output], Code).
```

Generates:

```perl
#!/usr/bin/env perl
use strict;
use warnings;
use JSON;

sub parent {
    my $callback = shift;
    my @facts = (
        ["alice", "bob"],
        ["bob", "charlie"]
    );
    foreach my $fact (@facts) {
        $callback->(@$fact);
    }
}

# JSON output wrapper
sub parent_json {
    my @results;
    parent(sub {
        push @results, [$_[1], $_[2]];
    });
    print encode_json(\@results);
}

# Run if executed directly
parent_json() unless caller;
```

Run directly:
```bash
perl parent.pl
# Output: [["alice","bob"],["bob","charlie"]]
```

## Pipeline Mode

The `pipeline` option generates shell-friendly executables:

```prolog
?- compile_predicate_to_perl(transform/2, [pipeline], Code).
```

Use in pipelines:
```bash
cat input.json | perl transform.pl | jq .
```

## FFI Bindings

The Perl target includes 150+ FFI bindings for common operations:

### String Operations
- `length/2`, `substr/4`, `index/3`, `rindex/3`
- `uc/2`, `lc/2`, `ucfirst/2`, `lcfirst/2`
- `split/3`, `join/3`, `reverse/2`
- `chomp/2`, `chop/2`, `trim/2`

### Array Operations
- `push/3`, `pop/2`, `shift/2`, `unshift/3`
- `sort/2`, `reverse/2`, `unique/2`
- `map/3`, `grep/3`, `reduce/3`

### Hash Operations
- `keys/2`, `values/2`, `each/3`
- `exists/3`, `delete/3`

### I/O Operations
- `print/1`, `say/1`, `warn/1`
- `open/3`, `close/1`, `read_file/2`
- `getline/2`, `slurp/2`

### Math Operations
- Arithmetic: `+/3`, `-/3`, `*/3`, `//3`, `%/3`
- Functions: `abs/2`, `sqrt/2`, `sin/2`, `cos/2`, `log/2`, `exp/2`
- Rounding: `int/2`, `floor/2`, `ceil/2`, `round/2`

## Performance Characteristics

- **Memory**: O(n) where n is the number of unique output tuples (for deduplication)
- **Time**: Nested-loop joins; adequate for modest data sets
- **Streaming**: Results are emitted immediately via callback
- **Memoization**: O(1) lookup for repeated recursive calls

## Limitations

- Inner joins only (no outer join patterns)
- String-based comparison (`eq`) by default
- No window functions
- No GROUP BY (only global aggregations)

## Usage Examples

### Command-line Execution

```bash
# Generate and run
swipl -g "compile_predicate_to_perl(ancestor/2, [json_output], C), format('~s', [C]), halt" > ancestor.pl
perl ancestor.pl
```

### Pipeline Integration

```bash
# Chain with jq
perl parent.pl | jq '.[] | {parent: .[0], child: .[1]}'

# Filter results
perl ancestor.pl | jq '[.[] | select(.[0] == "alice")]'
```

### Module Usage

```perl
use strict;
use warnings;
require 'predicates.pl';

# Use as library
my @results;
ancestor(sub { push @results, [@_]; });
print "Found " . scalar(@results) . " ancestor relationships\n";
```

## See Also

- [Ruby Target](ruby.md) - Similar CPS-based scripting target
- [Bash Target](bash.md) - Shell script generation
- [Target Overview](overview.md) - Comparison of all targets

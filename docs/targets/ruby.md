# Ruby Target

The Ruby target (`target(ruby)`) generates Ruby methods from Prolog predicates using block-based continuation-passing style. It provides idiomatic Ruby code with `yield` and block syntax, suitable for data processing and pipeline integration.

## Overview

Ruby programs use blocks to stream results, leveraging Ruby's native iterator pattern. This produces natural, readable code that integrates well with Ruby applications.

```prolog
% Compile to Ruby
?- compile_predicate_to_ruby(my_predicate/2, [], RubyCode).

% With JSON output wrapper
?- compile_predicate_to_ruby(my_predicate/2, [json_output], RubyCode).
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Facts | ✅ | Direct fact compilation to arrays with `each` |
| Single Rules | ✅ | Body-to-code translation with blocks |
| Multiple Rules (OR) | ✅ | Sequential clause evaluation |
| Semi-naive Recursion | ✅ | Datalog-style fixpoint with `Set` deduplication |
| Tail Recursion | ✅ | Accumulator patterns optimized to `loop` |
| Linear Recursion | ✅ | Memoized recursive calls with `@memo` |
| Joins | ✅ | Inner joins with `next unless` guards |
| Aggregations | ✅ | count, sum, min, max, avg |
| JSON Output | ✅ | JSON wrapper with `to_json` |
| Pipeline Mode | ✅ | stdin/stdout JSON streaming |

## Compilation Options

| Option | Description |
|--------|-------------|
| `json_output` | Generate `_json` method with JSON output |
| `json_input` | Read input facts from JSON (planned) |
| `pipeline` | Generate `run_pipeline` for shell integration |

## Block-Based CPS

All predicates generate methods that yield results to blocks:

```ruby
def parent
  facts = [
    ["alice", "bob"],
    ["bob", "charlie"]
  ]
  facts.each { |fact| yield(*fact) }
end

# Usage: stream all parent relationships
parent { |x, y| puts "#{x} -> #{y}" }
```

## Join Handling

When rules contain multiple goals sharing variables, the compiler generates proper join conditions:

```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

Generates:

```ruby
def grandparent
  parent do |p0, p1|
    parent do |p2, p3|
      next unless p2 == p1  # Join condition
      yield(p0, p3)
    end
  end
end
```

Key implementation details:
- **Unique parameter names** (`p0`, `p1`, etc.) avoid variable shadowing in nested blocks
- **Join conditions** (`next unless`) enforce variable equality between goals
- **Immediate projection** yields only the head variables

## Recursion Patterns

### Semi-naive Recursion (Datalog-style)

For transitive closure patterns, the compiler generates semi-naive iteration with Ruby's `Set`:

```prolog
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

Generates:

```ruby
require 'set'

def ancestor
  delta = []
  seen = Set.new

  # Base cases - seed the worklist
  parent do |p0, p1|
    key = [p0, p1]
    unless seen.include?(key)
      seen.add(key)
      delta << key
    end
  end

  # Semi-naive iteration
  until delta.empty?
    item = delta.shift
    yield(*item)

    parent do |p2, p3|
      next unless p3 == item[0]
      key = [p2, item[1]]
      unless seen.include?(key)
        seen.add(key)
        delta << key
      end
    end
  end
end
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

```ruby
def factorial(arg1, arg2, arg3, &block)
  # Tail recursion optimized to loop
  loop do
    if arg1 == 0
      block.call(arg2)
      break
    end
    tmp0 = (arg1 - 1)
    tmp1 = (arg2 * arg1)
    arg1 = tmp0
    arg2 = tmp1
  end
end
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

```ruby
def fib(arg1, &block)
  @memo ||= {}
  key = [arg1]
  if @memo.key?(key)
    return block.call(@memo[key])
  end

  if arg1 == 0
    @memo[key] = 0
    return block.call(0)
  end
  if arg1 == 1
    @memo[key] = 1
    return block.call(1)
  end

  tmp0 = (arg1 - 1)
  tmp1 = (arg1 - 2)
  fib(tmp0) do |r0|
    fib(tmp1) do |r1|
      result = (r0 + r1)
      @memo[key] = result
      block.call(result)
    end
  end
end
```

## Aggregations

The Ruby target supports `aggregate_all/3` with common aggregation templates:

```prolog
total_price(Total) :-
    aggregate_all(sum(Price), item(_, Price), Total).
```

Generates:

```ruby
def total_price(&block)
  agg_result = 0

  item do |g1, g2|
    tmpl_val = g2
    agg_result += tmpl_val
  end

  block.call(agg_result)
end
```

### Supported Aggregation Templates

| Template | Init | Update | Description |
|----------|------|--------|-------------|
| `count` | `0` | `+= 1` | Count matching tuples |
| `sum(X)` | `0` | `+= X` | Sum of values |
| `min(X)` | `nil` | `= X if nil or X <` | Minimum value |
| `max(X)` | `nil` | `= X if nil or X >` | Maximum value |
| `avg(X)` | `[0, 0]` | `sum += X; count += 1` | Average (sum/count) |

## JSON Output Mode

The `json_output` option generates an executable wrapper:

```prolog
?- compile_predicate_to_ruby(parent/2, [json_output], Code).
```

Generates:

```ruby
#!/usr/bin/env ruby
require 'set'
require 'json'

def parent
  facts = [
    ["alice", "bob"],
    ["bob", "charlie"]
  ]
  facts.each { |fact| yield(*fact) }
end

# JSON output wrapper
def parent_json
  results = []
  parent { |*args| results << [args[1], args[2]] }
  puts results.to_json
end

# Run if executed directly
parent_json if __FILE__ == $0
```

Run directly:
```bash
ruby parent.rb
# Output: [["alice","bob"],["bob","charlie"]]
```

## Pipeline Mode

The `pipeline` option generates shell-friendly executables:

```prolog
?- compile_predicate_to_ruby(transform/2, [pipeline], Code).
```

Use in pipelines:
```bash
cat input.json | ruby transform.rb | jq .
```

## FFI Bindings

The Ruby target includes 150+ FFI bindings for common operations:

### String Operations
- `length/2`, `size/2`, `bytesize/2`
- `upcase/2`, `downcase/2`, `capitalize/2`, `swapcase/2`
- `split/3`, `join/3`, `reverse/2`
- `strip/2`, `lstrip/2`, `rstrip/2`, `chomp/2`
- `gsub/4`, `sub/4`, `tr/4`

### Array Operations
- `push/3`, `pop/2`, `shift/2`, `unshift/3`
- `first/2`, `last/2`, `take/3`, `drop/3`
- `sort/2`, `sort_by/3`, `reverse/2`, `uniq/2`
- `map/3`, `select/3`, `reject/3`, `reduce/4`
- `flatten/2`, `compact/2`, `zip/3`

### Hash Operations
- `keys/2`, `values/2`, `each/3`
- `fetch/4`, `store/4`, `delete/3`
- `merge/3`, `slice/3`, `transform_values/3`

### I/O Operations
- `puts/1`, `print/1`, `p/1`
- `gets/1`, `readline/1`
- `File.read/2`, `File.write/3`, `File.readlines/2`
- `open/3`, `each_line/2`

### Math Operations
- Arithmetic: `+/3`, `-/3`, `*/3`, `//3`, `%/3`, `**/3`
- Functions: `abs/2`, `sqrt/2`, `sin/2`, `cos/2`, `log/2`, `exp/2`
- Rounding: `to_i/2`, `floor/2`, `ceil/2`, `round/2`
- Comparison: `min/3`, `max/3`, `between?/4`

### Type Conversion
- `to_s/2`, `to_i/2`, `to_f/2`, `to_a/2`, `to_h/2`
- `to_sym/2`, `inspect/2`

## Performance Characteristics

- **Memory**: O(n) where n is the number of unique output tuples (for deduplication)
- **Time**: Nested-loop joins; adequate for modest data sets
- **Streaming**: Results are yielded immediately via blocks
- **Memoization**: O(1) lookup for repeated recursive calls

## Limitations

- Inner joins only (no outer join patterns)
- No window functions
- No GROUP BY (only global aggregations)
- Requires Ruby 2.0+ for keyword arguments

## Usage Examples

### Command-line Execution

```bash
# Generate and run
swipl -g "compile_predicate_to_ruby(ancestor/2, [json_output], C), format('~s', [C]), halt" > ancestor.rb
ruby ancestor.rb
```

### Pipeline Integration

```bash
# Chain with jq
ruby parent.rb | jq '.[] | {parent: .[0], child: .[1]}'

# Filter results
ruby ancestor.rb | jq '[.[] | select(.[0] == "alice")]'
```

### Module Usage

```ruby
require_relative 'predicates'

# Use as library
results = []
ancestor { |*args| results << args }
puts "Found #{results.size} ancestor relationships"
```

### Rails Integration

```ruby
# In a Rails model or service
class AncestorService
  def self.find_all
    results = []
    ancestor { |x, y| results << { ancestor: x, descendant: y } }
    results
  end
end
```

## Ruby-Specific Idioms

The generated code follows Ruby conventions:

| Prolog Pattern | Ruby Idiom |
|----------------|------------|
| Facts iteration | `array.each { \|item\| yield(*item) }` |
| Join condition | `next unless condition` |
| Deduplication | `Set.new` with `include?` / `add` |
| Memoization | `@memo \|\|= {}` instance variable |
| Loop iteration | `loop do ... break ... end` |
| Block passing | `&block` parameter with `block.call` |

## See Also

- [Perl Target](perl.md) - Similar CPS-based scripting target
- [Python Target](../targets/comparison.md) - Alternative scripting target
- [Target Overview](overview.md) - Comparison of all targets

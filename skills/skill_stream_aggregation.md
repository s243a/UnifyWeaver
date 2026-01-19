# Skill: Stream Aggregation

Runtime aggregation within generated code for Go, C#, Perl, and Ruby targets.

## When to Use

- User asks "how do I aggregate data at runtime?"
- User wants COUNT, SUM, AVG in generated code (not SQL)
- User asks about `aggregate_all` in stream targets
- User needs HAVING-style filtering after aggregation
- User wants to group and accumulate values

## Quick Start

### Basic Aggregation

```prolog
% Count all items
total_count(Count) :-
    aggregate_all(count, item(_), Count).

% Sum values
total_price(Total) :-
    aggregate_all(sum(Price), order(_, Price), Total).
```

### With Grouping

```prolog
% Count per category
category_count(Category, Count) :-
    aggregate_all(count, item(_, Category), Category, Count).

% Total sales per region
regional_sales(Region, Total) :-
    aggregate_all(sum(Amount), sale(Region, _, Amount), Region, Total).
```

## Aggregation Operators

| Operator | Description | Result Type |
|----------|-------------|-------------|
| `count` | Count matching items | integer |
| `sum(V)` | Sum of values | numeric |
| `min(V)` | Minimum value | same as V |
| `max(V)` | Maximum value | same as V |
| `avg(V)` | Average value | float |
| `set(V)` | Unique values | set/list |
| `bag(V)` | All values (with duplicates) | list |

## Target Support

| Target | `aggregate_all/3` | `aggregate_all/4` | HAVING |
|--------|-------------------|-------------------|--------|
| **Go** | Full | Full | Yes |
| **C#** | Full | Full | Partial |
| **Perl** | Basic | Partial | No |
| **Ruby** | Basic | Partial | No |

## Go Target

The Go target has the most complete aggregation support.

### Generated Code Pattern

```prolog
% Prolog
dept_headcount(Dept, Count) :-
    aggregate_all(count, employee(_, Dept, _), Dept, Count).
```

```go
// Generated Go
func deptHeadcount() map[string]int {
    result := make(map[string]int)
    for _, emp := range employees {
        result[emp.Dept]++
    }
    return result
}
```

### HAVING Clause

Filter groups after aggregation:

```prolog
% Departments with more than 5 employees
large_depts(Dept, Count) :-
    aggregate_all(count, employee(_, Dept, _), Dept, Count),
    Count > 5.
```

```go
// Generated Go with HAVING filter
func largeDepts() map[string]int {
    result := make(map[string]int)
    for _, emp := range employees {
        result[emp.Dept]++
    }
    // HAVING filter
    for dept, count := range result {
        if count <= 5 {
            delete(result, dept)
        }
    }
    return result
}
```

## C# Target

C# supports aggregation in both query runtime and generator modes.

### Query Runtime Mode

Uses semi-naive evaluation for efficient incremental aggregation:

```prolog
% Prolog
total_sales(Total) :-
    aggregate_all(sum(Amount), sale(_, Amount), Total).
```

```csharp
// Generated C# (query runtime)
var total = facts
    .Where(f => f.Predicate == "sale")
    .Sum(f => (decimal)f.Args[1]);
```

### Generator Mode

```prolog
% With grouping
category_total(Category, Total) :-
    aggregate_all(sum(Price), product(_, Category, Price), Category, Total).
```

```csharp
// Generated C# (generator mode)
var categoryTotals = products
    .GroupBy(p => p.Category)
    .ToDictionary(g => g.Key, g => g.Sum(p => p.Price));
```

## Perl Target

Basic aggregation support:

```prolog
% Prolog
word_count(Count) :-
    aggregate_all(count, word(_), Count).
```

```perl
# Generated Perl
my $count = scalar(@words);
```

## Ruby Target

Basic aggregation support:

```prolog
% Prolog
total_amount(Total) :-
    aggregate_all(sum(Amount), payment(_, Amount), Total).
```

```ruby
# Generated Ruby
total = payments.sum { |p| p[:amount] }
```

## Database Integration

Each target integrates with its embedded database:

| Target | Database | Aggregation Location |
|--------|----------|---------------------|
| Go | BBolt | In-memory after fetch |
| C# | LiteDB | LINQ on collections |
| Perl | - | In-memory |
| Ruby | - | In-memory |

### Go + BBolt Example

```prolog
% Aggregate from stored data
stored_stats(Category, Avg) :-
    fetch_from_db(items, Items),
    aggregate_all(avg(Price), member(item(_, Category, Price), Items), Category, Avg).
```

## Commands

### Compile Go with Aggregation
```bash
swipl -g "use_module('src/unifyweaver/targets/go_target'), \
          compile_to_go(my_aggregation/2, [], Code), \
          write(Code)" -t halt my_rules.pl
```

### Compile C# with Query Runtime
```bash
swipl -g "use_module('src/unifyweaver/targets/csharp_target'), \
          compile_query_runtime(my_aggregate/1, Code), \
          write(Code)" -t halt my_rules.pl
```

## Best Practices

### Choose the Right Operator

```prolog
% Use set/1 for unique values
unique_categories(Categories) :-
    aggregate_all(set(Cat), product(_, Cat, _), Categories).

% Use bag/1 when duplicates matter
all_prices(Prices) :-
    aggregate_all(bag(Price), order(_, Price), Prices).
```

### Combine with Filtering

```prolog
% Pre-filter before aggregation
active_user_count(Count) :-
    aggregate_all(count, (user(Id, _), is_active(Id)), Count).

% Post-filter with HAVING (Go target)
popular_items(Item, Sales) :-
    aggregate_all(count, purchase(_, Item), Item, Sales),
    Sales >= 100.
```

## Related

**Parent Skill:**
- `skill_aggregation_patterns.md` - Overview of aggregation approaches

**Other Skills:**
- `skill_sql_target.md` - Declarative SQL aggregation
- `skill_fuzzy_search.md` - Score-based aggregation

**Documentation:**
- `docs/BINDING_MATRIX.md` - Target feature matrix

**Code:**
- `src/unifyweaver/targets/go_target.pl` - Go aggregation implementation
- `src/unifyweaver/targets/csharp_target.pl` - C# aggregation support
- `src/unifyweaver/targets/perl_target.pl` - Perl aggregation
- `src/unifyweaver/targets/ruby_target.pl` - Ruby aggregation

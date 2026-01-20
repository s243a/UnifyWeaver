# Skill: Data Binding

Generate React hooks, providers, and WebSocket sync for binding Prolog data to UI components.

## When to Use

- User asks "how do I connect Prolog data to React?"
- User needs reactive UI updates
- User wants real-time WebSocket synchronization
- User needs two-way data binding for forms

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/data_binding_generator').

% Define a data source
data_source(sales, [
    predicate(sales_record/4),
    fields([date, product, quantity, amount]),
    primary_key(id)
]).

% Create binding to chart
binding(sales_chart, sales, [
    x_axis(date),
    y_axis(amount),
    series(product)
]).

% Generate React hook
generate_binding_hook(sales_chart, HookCode).
```

## Data Sources

### Basic Data Source

```prolog
data_source(Name, Options).
```

| Option | Type | Description |
|--------|------|-------------|
| `predicate(P/A)` | indicator | Source predicate |
| `fields(List)` | list | Field names |
| `primary_key(K)` | atom | Primary key field |
| `refresh_interval(Ms)` | integer | Polling interval (0 = no polling) |
| `cache(Bool)` | boolean | Enable caching |
| `sort_by(Field, Dir)` | atom, atom | Default sort |

### Pre-defined Sources

| Source | Fields | Purpose |
|--------|--------|---------|
| `default` | id, value | Generic data |
| `time_series` | timestamp, series, value | Time-based data |
| `graph_data` | nodes, edges | Network graphs |
| `hierarchical` | id, parent_id, label | Tree structures |

### Custom Source Example

```prolog
data_source(inventory, [
    predicate(inventory_item/5),
    fields([sku, name, quantity, price, category]),
    primary_key(sku),
    sort_by(name, asc),
    cache(true)
]).
```

## Computed Sources

Derived data from aggregations or transformations:

```prolog
computed_source(Name, Options).
```

### Aggregation

```prolog
computed_source(sales_summary, [
    base_source(sales),
    computation(aggregate),
    group_by([product]),
    aggregations([
        sum(amount, total_sales),
        avg(amount, avg_sale),
        count(_, sale_count)
    ]),
    cache(true)
]).
```

### Filtering

```prolog
computed_source(high_value, [
    base_source(sales),
    computation(filter),
    filter_predicate(amount > 1000),
    cache(false)
]).
```

### Joins

```prolog
computed_source(product_sales, [
    sources([products, sales]),
    computation(join),
    join_on(product_id),
    join_type(inner)
]).
```

## Bindings

### One-Way Binding

```prolog
binding(Component, Source, Mapping).
```

```prolog
binding(line_chart, time_series, [
    x_axis(timestamp),
    y_axis(value),
    series(series),
    transform(none)
]).

binding(bar_chart, aggregated, [
    x_axis(category),
    y_axis(total),
    color(category)
]).

binding(network_graph, graph_data, [
    nodes(graph_node),
    edges(graph_edge),
    node_id(id),
    node_label(label),
    edge_source(source),
    edge_target(target)
]).
```

### Two-Way Binding

For editable components:

```prolog
two_way_binding(Component, Source, Mapping).
```

```prolog
two_way_binding(data_table, sales, [
    columns([date, product, quantity, amount]),
    editable([quantity, amount]),
    on_edit(update_record),
    validation([
        field(amount, number),
        field(amount, min(0))
    ])
]).

two_way_binding(form, user_profile, [
    fields([name, email, phone]),
    editable([name, email, phone]),
    on_change(update_field),
    debounce(300)
]).
```

## Code Generation

### Generate React Hook

```prolog
generate_binding_hook(Component, HookCode).
```

**Output (static data):**
```typescript
import { useState, useEffect, useCallback } from "react";

interface SalesData {
  date: Date | string;
  product: string;
  quantity: number;
  amount: number;
}

interface UseSalesChartDataResult {
  data: SalesData[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export const useSalesChartData = (): UseSalesChartDataResult => {
  const [data, setData] = useState<SalesData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/data/sales");
      const rawData = await response.json();
      setData(rawData);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};
```

**Output (polling):**
Adds `isPolling`, `startPolling`, `stopPolling` to the hook.

### Generate Data Provider

```prolog
generate_data_provider(Source, ProviderCode).
```

Generates React Context provider with CRUD operations:
- `data` - Current data array
- `loading` - Loading state
- `error` - Error state
- `refetch()` - Refresh data
- `updateRecord(id, updates)` - Update record
- `deleteRecord(id)` - Delete record
- `addRecord(record)` - Add record

### Generate WebSocket Sync

```prolog
generate_websocket_sync(Source, SyncCode).
```

**Features:**
- Real-time updates via WebSocket
- Auto-reconnect on disconnect
- Event types: `sync`, `insert`, `update`, `delete`
- Send mutations to server

**Output:**
```typescript
export const useSalesSync = (wsUrl?: string): WebSocketSyncResult => {
  // WebSocket connection management
  // Real-time data synchronization
  // Reconnection logic
  return { data, connected, error, send, reconnect };
};
```

### Generate Mutation Handler

```prolog
generate_mutation_handler(Source, HandlerCode).
```

**Output:**
```typescript
export const useSalesMutations = (onSuccess?: () => void): MutationHandlers => {
  return {
    create: async (data) => { /* POST /api/data/sales */ },
    update: async (id, data) => { /* PATCH /api/data/sales/:id */ },
    remove: async (id) => { /* DELETE /api/data/sales/:id */ },
    state: { loading, error }
  };
};
```

### Generate Computed Hook

```prolog
generate_computed_hook(Source, HookCode).
```

For computed sources, generates `useMemo`-based transformation:

```typescript
export const useSalesSummary = (sourceData: SalesData[]): SalesSummaryData[] => {
  return useMemo(() => {
    // Grouping and aggregation logic
  }, [sourceData]);
};
```

### Generate Types

```prolog
generate_binding_types(Source, TypesCode).
```

**Output:**
```typescript
export interface SalesData {
  date: Date | string;
  product: string;
  quantity: number;
  amount: number;
}
```

## Type Inference

Fields are automatically typed based on naming conventions:

| Pattern | TypeScript Type |
|---------|-----------------|
| `*id`, `*name`, `*label` | `string` |
| `*count`, `*amount`, `*value` | `number` |
| `*date`, `*time`, `*timestamp` | `Date \| string` |
| `is_*`, `has_*`, `*enabled` | `boolean` |
| `properties` | `Record<string, unknown>` |
| Default | `unknown` |

Override with:
```prolog
infer_field_type(my_field, 'CustomType').
```

## Query Generation

### Fetch Query

```prolog
generate_fetch_query(Source, Query).
% Query = 'fetch("/api/data/sales?sort=date&order=asc")'
```

### Subscribe Query

```prolog
generate_subscribe_query(Source, Query).
% Query = 'ws.send(JSON.stringify({ type: "subscribe", source: "sales" }))'
```

### Update Mutation

```prolog
generate_update_mutation(Source, Mutation).
```

## Management

```prolog
% Declare new source
declare_data_source(my_source, [...]).

% Declare new binding
declare_binding(my_component, my_source, [...]).

% Clear all bindings
clear_bindings.
```

## Common Patterns

### Dashboard with Multiple Charts

```prolog
% Define sources
data_source(metrics, [fields([time, cpu, memory, disk])]).
computed_source(hourly_avg, [
    base_source(metrics),
    computation(aggregate),
    group_by([hour(time)]),
    aggregations([avg(cpu, avg_cpu), avg(memory, avg_memory)])
]).

% Bindings for each chart
binding(cpu_chart, metrics, [x_axis(time), y_axis(cpu)]).
binding(memory_chart, metrics, [x_axis(time), y_axis(memory)]).
binding(summary_chart, hourly_avg, [x_axis(hour), y_axis(avg_cpu)]).

% Generate all hooks
generate_binding_hook(cpu_chart, CPUHook).
generate_binding_hook(memory_chart, MemoryHook).
generate_binding_hook(summary_chart, SummaryHook).
```

### Editable Data Table

```prolog
data_source(users, [
    fields([id, name, email, role]),
    primary_key(id)
]).

two_way_binding(user_table, users, [
    columns([name, email, role]),
    editable([name, email, role]),
    on_edit(update_user)
]).

generate_binding_hook(user_table, TableHook).
generate_mutation_handler(users, MutationHook).
```

## Related

**Parent Skill:**
- `skill_gui_runtime.md` - GUI runtime sub-master

**Sibling Skills:**
- `skill_webassembly.md` - LLVM/WASM
- `skill_browser_python.md` - Pyodide

**Code:**
- `src/unifyweaver/glue/data_binding_generator.pl` - Main generator

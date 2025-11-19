# Skill: Declaring JSON Dynamic Sources

Use this skill whenever a playbook instructs you to read JSON data via `source/3`.

## Column projection mode
- Always provide `columns/1` so the arity matches the number of projected fields.
- Column names support dot notation and array indices (e.g., `items[0].total`).
- Example:
  ```prolog
  :- source(json, order_totals, [
      json_file('data/orders.json'),
      columns(['order.customer.name', 'items[0].product', 'items[0].total'])
  ]).
  ```

## Return-object mode
- Use when you want full JSON objects as rows.
- Requirements:
  - `arity(1)`
  - `return_object(true)`
  - `type_hint/1` specifying a .NET type (e.g., `'System.Text.Json.Nodes.JsonObject, System.Text.Json'`).
- Example:
  ```prolog
  :- source(json, raw_products, [
      json_file('test_data/test_products.json'),
      arity(1),
      return_object(true),
      type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json')
  ]).
  ```

## Validation expectations
- Missing `columns/1` (when `return_object(false)`) causes a `domain_error(json_columns, _)`.
- Column count must equal predicate arity.
- Empty column names or malformed selectors raise `domain_error(json_column_entry, _)`.
- `return_object(true)` without a `type_hint/1` or with arity ≠ 1 raises an error.

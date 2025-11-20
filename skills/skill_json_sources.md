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

## JSONPath selectors
- Use `jsonpath('$.orders[*].total')` to select data when simple dot paths are insufficient.
- Supported features: root `$`, dotted properties, bracket properties (`['foo']`), array indices `[0]`, wildcards `[*]`, and recursive descent `..name`.
- Works in both `columns/1` and `schema/1`. Wildcards currently return the first matching value.
- Strings that already begin with `$` are automatically treated as JSONPath selectors, so `columns(['$.orders[0].id'])` also works.

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

## Schema-generated records
- Use `schema/1` to declare a typed record and have the C# backend generate a POCO.
- Requirements:
  - `schema([field(Name, Path, Type), ...])`
  - Optional `record_type('ProductRecord')` to name the generated record.
  - Predicate arity must be `1`; `return_object(true)` is implied.
- Example:
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
  % yields ProductRecord { Id = P001, Name = Laptop, Price = 999 }
  ```

## Nested schema records
- Return structured sub-objects by using `record(TypeName, Fields)` (or `record(Fields)` to auto-name from the field).
- Nested selectors are evaluated relative to the selected sub-object; JSONPath and dot notation both work.
- Example:
  ```prolog
  :- source(json, order_rows, [
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
  ```
- Generated C# includes `OrderRecord`, `LineItemRecord`, and `OrderSummaryRecord`; runtime instantiates nested POCOs automatically.

## Validation expectations
- Missing `columns/1` (when `return_object(false)`) causes a `domain_error(json_columns, _)`.
- Column count must equal predicate arity.
- Empty column names or malformed selectors raise `domain_error(json_column_entry, _)`.
- `return_object(true)` without a `type_hint/1` or with arity ≠ 1 raises an error.

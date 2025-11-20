:- module(test_json_source_validation, []).

:- use_module(library(plunit)).
:- use_module('src/unifyweaver/sources').

:- begin_tests(json_source_validation).

test(missing_columns,
     [error(domain_error(json_columns, _))]) :-
    source(json, missing_cols,
        [ json_file('test_data/test_products.json'),
          arity(2)
        ]).

test(columns_arity_mismatch,
     [error(domain_error(json_columns_arity, _))]) :-
    source(json, mismatch_cols,
        [ json_file('test_data/test_products.json'),
          columns([id]),
          arity(2)
        ]).

test(return_object_missing_type,
     [error(domain_error(json_type_hint, _))]) :-
    source(json, missing_type,
        [ json_file('test_data/test_products.json'),
          return_object(true),
          arity(1)
        ]).

test(return_object_wrong_arity,
     [error(domain_error(json_return_object_arity, _))]) :-
    source(json, wrong_return_arity,
        [ json_file('test_data/test_products.json'),
          type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json'),
          return_object(true),
          arity(2)
        ]).

test(return_object_with_columns,
     [error(domain_error(json_return_object_columns, _))]) :-
    source(json, return_with_cols,
        [ json_file('test_data/test_products.json'),
          type_hint('System.Text.Json.Nodes.JsonObject, System.Text.Json'),
          return_object(true),
          arity(1),
          columns([id])
        ]).

test(schema_requires_arity_one,
     [error(domain_error(json_schema_arity, _))]) :-
    source(json, schema_wrong_arity,
        [ json_file('test_data/test_products.json'),
          schema([field(id, 'id', string)]),
          arity(2)
        ]).

test(schema_conflicts_with_columns,
     [error(domain_error(json_schema_columns_conflict, _))]) :-
    source(json, schema_columns_conflict,
        [ json_file('test_data/test_products.json'),
          schema([field(id, 'id', string)]),
          columns([id])
        ]).

:- end_tests(json_source_validation).

:- initialization(main).

main :-
    run_tests,
    halt.

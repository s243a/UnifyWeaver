:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_typescript_json_schema.pl - plunit suite for JSON schema mode in the
% TypeScript/Node data-source consumer (G-P9 schema mode). Exercises the schema
% path added to json_source.pl (the `_typescript_schema` templates + schema
% projection helpers) and the routing added to typescript_source_compiler.pl:
% a JSON source declared with schema([field(Name, Path, Type), ...]) compiles to
% a self-contained Node script (fs + JSON.parse, no npm deps) that parses each
% JSON record into a TYPED object with exactly the schema's field keys, in
% declared order, coercing each field to its declared type, and emits each record
% as a JSON.stringify line. Includes a control test proving a non-schema (flat
% columns) JSON source still emits the old flat ':'-joined output unchanged.
%
% Isolated from tests/core/test_typescript_source.pl (which has changes on another
% branch) - all schema-mode tests live here.
%
% Run: swipl -q -g test_typescript_json_schema -t halt tests/core/test_typescript_json_schema.pl

:- module(test_typescript_json_schema, [test_typescript_json_schema/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module('../../src/unifyweaver/sources').
:- use_module('../../src/unifyweaver/targets/typescript_target').
:- use_module('../../src/unifyweaver/core/dynamic_source_compiler').

test_typescript_json_schema :-
    run_tests([typescript_json_schema]).

:- begin_tests(typescript_json_schema).

has(Code, Substr) :- once(sub_string(Code, _, _, _, Substr)).

% node availability gate (>= 22, --experimental-strip-types)
node_available :-
    catch(( process_create(path(node), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

% Write Code to a temp .ts file, run under node --experimental-strip-types with
% Argv, return raw stdout (each row on its own line preserved).
ts_write_run_lines(Code, Argv, Lines) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.ts', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        ts_node_exec_lines(File, Argv, Lines),
        catch(delete_file(File), _, true)).

ts_node_exec_lines(File, Argv, Lines) :-
    append(['--experimental-strip-types', File], Argv, Args),
    process_create(path(node), Args,
                   [stdout(pipe(O)), stderr(null), process(P)]),
    read_string(O, _, Str), close(O), process_wait(P, _),
    split_string(Str, "\n", "\n \t\r", Parts0),
    exclude(==(""), Parts0, Lines).

abs(Rel, Abs) :- once(absolute_file_name(Rel, Abs)).

% JSON fixture with native-typed values (price is a JSON number).
write_typed_json_fixture(File) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.json', File),
    open(File, write, W),
    write(W, '[\n'),
    write(W, '  {"id": "P001", "name": "Laptop", "price": 999, "active": true},\n'),
    write(W, '  {"id": "P002", "name": "Mouse", "price": 25, "active": false},\n'),
    write(W, '  {"id": "P003", "name": "Keyboard", "price": 75, "active": true}\n'),
    write(W, ']\n'),
    close(W).

% JSON fixture where numeric/boolean fields arrive as STRINGS, to prove coercion.
write_stringy_json_fixture(File) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.json', File),
    open(File, write, W),
    write(W, '[\n'),
    write(W, '  {"id": "P001", "qty": "10", "rate": "1.5", "flag": "true"},\n'),
    write(W, '  {"id": "P002", "qty": "3", "rate": "2.25", "flag": "false"}\n'),
    write(W, ']\n'),
    close(W).

% ============================================================================
% Schema mode: emitted code shape
% ============================================================================

% A declared schema JSON source routes through the schema path and emits typed
% record construction (no flat ':'-join, no columns projection).
test(schema_compiles_to_typed_node_script,
        [setup(cleanup_schema), cleanup(cleanup_schema)]) :-
    write_typed_json_fixture(JF),
    once(source(json, ts_schema_prod, [ json_file(JF),
        schema([ field(id, id, string),
                 field(name, name, string),
                 field(price, price, integer) ]) ])),
    once(is_dynamic_source(ts_schema_prod/1)),
    once(typescript_target:compile_predicate_to_typescript(ts_schema_prod/1, [], Code)),
    has(Code, 'require("fs")'),
    has(Code, 'JSON.parse'),
    % schema descriptor array in declared order, with types
    has(Code, '{"key":"id","path":"id","type":"string"}'),
    has(Code, '{"key":"name","path":"name","type":"string"}'),
    has(Code, '{"key":"price","path":"price","type":"integer"}'),
    % typed-object output shape (JSON.stringify of a per-record object)
    has(Code, 'JSON.stringify'),
    has(Code, 'ts_schema_prod_coerce'),
    % NOT the flat projection path
    \+ has(Code, '.join(":")'),
    \+ has(Code, 'jq'),
    catch(delete_file(JF), _, true).

% Declared field order is preserved in the schema descriptor array even when it
% differs from the object key order.
test(schema_field_order_preserved,
        [setup(cleanup_schema), cleanup(cleanup_schema)]) :-
    write_typed_json_fixture(JF),
    once(source(json, ts_schema_prod, [ json_file(JF),
        schema([ field(price, price, integer),
                 field(id, id, string),
                 field(name, name, string) ]) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_schema_prod/1, [], Code)),
    % price precedes id precedes name in the descriptor array
    once(sub_string(Code, PP, _, _, '"key":"price"')),
    once(sub_string(Code, PI, _, _, '"key":"id"')),
    once(sub_string(Code, PN, _, _, '"key":"name"')),
    PP < PI, PI < PN,
    catch(delete_file(JF), _, true).

% ============================================================================
% Schema mode: node execution
% ============================================================================

% Native-typed values: each record is emitted as a JSON object with typed values
% (string id/name, numeric price, boolean active) in declared order.
test(schema_node_execution_typed,
        [ condition(node_available),
          setup(cleanup_schema), cleanup(cleanup_schema) ]) :-
    write_typed_json_fixture(JF),
    once(source(json, ts_schema_prod, [ json_file(JF),
        schema([ field(id, id, string),
                 field(name, name, string),
                 field(price, price, integer),
                 field(active, active, boolean) ]) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_schema_prod/1, [], Code)),
    ts_write_run_lines(Code, [], Lines),
    Lines == [ "{\"id\":\"P001\",\"name\":\"Laptop\",\"price\":999,\"active\":true}",
               "{\"id\":\"P002\",\"name\":\"Mouse\",\"price\":25,\"active\":false}",
               "{\"id\":\"P003\",\"name\":\"Keyboard\",\"price\":75,\"active\":true}" ],
    catch(delete_file(JF), _, true).

% Coercion: string-encoded numbers/booleans are coerced to JS Number/Boolean per
% the declared type (integer/float/boolean), not left as strings.
test(schema_node_execution_coercion,
        [ condition(node_available),
          setup(cleanup_schema), cleanup(cleanup_schema) ]) :-
    write_stringy_json_fixture(JF),
    once(source(json, ts_schema_prod, [ json_file(JF),
        schema([ field(id, id, string),
                 field(qty, qty, integer),
                 field(rate, rate, float),
                 field(flag, flag, boolean) ]) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_schema_prod/1, [], Code)),
    ts_write_run_lines(Code, [], Lines),
    Lines == [ "{\"id\":\"P001\",\"qty\":10,\"rate\":1.5,\"flag\":true}",
               "{\"id\":\"P002\",\"qty\":3,\"rate\":2.25,\"flag\":false}" ],
    catch(delete_file(JF), _, true).

cleanup_schema :-
    retractall(dynamic_source_compiler:dynamic_source_def(ts_schema_prod/_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(ts_schema_prod/_, _)).

% ============================================================================
% CONTROL: non-schema (flat columns) JSON source is unchanged
% ============================================================================

% A JSON source declared with columns([...]) (no schema) must still emit the old
% flat projection path: a JS projection array + ':'-join, and NOT the schema
% templates. Proves schema mode is additive.
test(control_flat_columns_unchanged_code,
        [setup(cleanup_flat), cleanup(cleanup_flat)]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_flat_prod, [ json_file(JF),
        columns([id, name, price]), arity(3) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_flat_prod/3, [], Code)),
    % flat projection path markers
    has(Code, '["id","name","price"]'),
    has(Code, '.join(":")'),
    % schema-mode markers must be absent
    \+ has(Code, '_coerce'),
    \+ has(Code, '{"key":"id"'),
    \+ has(Code, 'JSON.stringify(').

% And it still runs to the old flat ':'-joined rows.
test(control_flat_columns_unchanged_execution,
        [ condition(node_available),
          setup(cleanup_flat), cleanup(cleanup_flat) ]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_flat_prod, [ json_file(JF),
        columns([id, name, price]), arity(3) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_flat_prod/3, [], Code)),
    ts_write_run_lines(Code, [], Lines),
    Lines == ["P001:Laptop:999", "P002:Mouse:25", "P003:Keyboard:75"].

cleanup_flat :-
    retractall(dynamic_source_compiler:dynamic_source_def(ts_flat_prod/_, _, _)),
    retractall(dynamic_source_compiler:dynamic_source_metadata(ts_flat_prod/_, _)).

:- end_tests(typescript_json_schema).

:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% test_typescript_source.pl - plunit suite for the TypeScript/Node data-source
% consumer (G-P9). Exercises the routing clause added to typescript_target.pl
% and the typescript_source_compiler wrapper: a predicate declared as a JSON or
% CSV data source (via sources:source/3) compiles to a self-contained Node
% script (fs + JSON.parse, no npm deps) and, when node is available, runs over a
% fixture file and yields the expected rows.
%
% Run: swipl -q -g test_typescript_source -t halt tests/core/test_typescript_source.pl

:- module(test_typescript_source, [test_typescript_source/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module('../../src/unifyweaver/sources').
:- use_module('../../src/unifyweaver/targets/typescript_target').
:- use_module('../../src/unifyweaver/core/dynamic_source_compiler').

test_typescript_source :-
    run_tests([typescript_source]).

:- begin_tests(typescript_source).

has(Code, Substr) :- once(sub_string(Code, _, _, _, Substr)).

% node availability gate (>= 22, --experimental-strip-types)
node_available :-
    catch(( process_create(path(node), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

% Write Code to a temp .ts file, run under node --experimental-strip-types with
% Argv, return trimmed (space-normalised) stdout.
ts_write_run(Code, Argv, Out) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.ts', File),
    setup_call_cleanup(
        ( open(File, write, W), write(W, Code), close(W) ),
        ts_node_exec(File, Argv, Out),
        catch(delete_file(File), _, true)).

ts_node_exec(File, Argv, Out) :-
    append(['--experimental-strip-types', File], Argv, Args),
    process_create(path(node), Args,
                   [stdout(pipe(O)), stderr(null), process(P)]),
    read_string(O, _, Str), close(O), process_wait(P, _),
    normalize_space(atom(Out), Str).

abs(Rel, Abs) :- once(absolute_file_name(Rel, Abs)).

write_csv_fixture(File) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.csv', File),
    open(File, write, W),
    write(W, 'name,age,city\nalice,25,nyc\nbob,30,sf\ncharlie,35,la\n'),
    close(W).

% Tab-delimited fixture (same data, TAB separators).
write_tsv_fixture(File) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.tsv', File),
    open(File, write, W),
    write(W, 'name\tage\tcity\nalice\t25\tnyc\nbob\t30\tsf\ncharlie\t35\tla\n'),
    close(W).

% Pipe-delimited fixture (same data, '|' separators).
write_psv_fixture(File) :-
    tmp_file_stream(text, Base, S0), close(S0),
    atom_concat(Base, '.psv', File),
    open(File, write, W),
    write(W, 'name|age|city\nalice|25|nyc\nbob|30|sf\ncharlie|35|la\n'),
    close(W).

% ============================================================================
% JSON source -> Node
% ============================================================================

% A declared JSON source routes through the TS data-source path and emits a
% self-contained Node script (fs + JSON.parse, no npm deps, no jq).
test(json_compiles_to_node_script, [setup(cleanup_json), cleanup(cleanup_json)]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_product, [ json_file(JF), columns([id, name, price]), arity(3) ])),
    once(is_dynamic_source(ts_product/3)),
    once(typescript_target:compile_predicate_to_typescript(ts_product/3, [], Code)),
    has(Code, 'require(\"fs\")'),
    has(Code, 'JSON.parse'),
    has(Code, 'function ts_product'),
    \+ has(Code, 'jq').

% Run the emitted script and check the rows (values joined with ':').
test(json_node_execution, [ condition(node_available),
                            setup(cleanup_json), cleanup(cleanup_json) ]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_product, [ json_file(JF), columns([id, name, price]), arity(3) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_product/3, [], Code)),
    ts_write_run(Code, [], Out),
    Out == 'P001:Laptop:999 P002:Mouse:25 P003:Keyboard:75'.

cleanup_json :- retractall(dynamic_source_compiler:dynamic_source_def(ts_product/_, _, _)).

% ============================================================================
% CSV source -> Node
% ============================================================================

test(csv_compiles_to_node_script, [setup(cleanup_csv), cleanup(cleanup_csv)]) :-
    write_csv_fixture(CF),
    source(csv, ts_person, [ csv_file(CF), has_header(true) ]),
    once(is_dynamic_source(ts_person/3)),
    once(typescript_target:compile_predicate_to_typescript(ts_person/3, [], Code)),
    has(Code, 'require(\"fs\")'),
    has(Code, 'function ts_person'),
    has(Code, '.split('),
    catch(delete_file(CF), _, true).

% Run the emitted script: full stream, then key lookup on the first column.
test(csv_node_execution, [ condition(node_available),
                           setup(cleanup_csv), cleanup(cleanup_csv) ]) :-
    write_csv_fixture(CF),
    source(csv, ts_person, [ csv_file(CF), has_header(true) ]),
    once(typescript_target:compile_predicate_to_typescript(ts_person/3, [], Code)),
    ts_write_run(Code, [], OutAll),
    OutAll == 'alice:25:nyc bob:30:sf charlie:35:la',
    ts_write_run(Code, [bob], OutBob),
    OutBob == 'bob:30:sf',
    catch(delete_file(CF), _, true).

cleanup_csv :- retractall(dynamic_source_compiler:dynamic_source_def(ts_person/_, _, _)).

% ============================================================================
% CSV non-comma delimiters (G-P9 polish)
% ============================================================================

% Tab: the emitted script must split on a real tab ("\t"), not a comma.
test(csv_tab_in_code, [setup(cleanup_tsv), cleanup(cleanup_tsv)]) :-
    write_tsv_fixture(TF),
    source(csv, ts_tab, [ csv_file(TF), has_header(true), delimiter('\t') ]),
    once(is_dynamic_source(ts_tab/3)),
    once(typescript_target:compile_predicate_to_typescript(ts_tab/3, [], Code)),
    has(Code, '.split("\\t")'),
    catch(delete_file(TF), _, true).

test(csv_tab_node_execution, [ condition(node_available),
                               setup(cleanup_tsv), cleanup(cleanup_tsv) ]) :-
    write_tsv_fixture(TF),
    source(csv, ts_tab, [ csv_file(TF), has_header(true), delimiter('\t') ]),
    once(typescript_target:compile_predicate_to_typescript(ts_tab/3, [], Code)),
    ts_write_run(Code, [], OutAll),
    OutAll == 'alice:25:nyc bob:30:sf charlie:35:la',
    ts_write_run(Code, [bob], OutBob),
    OutBob == 'bob:30:sf',
    catch(delete_file(TF), _, true).

cleanup_tsv :- retractall(dynamic_source_compiler:dynamic_source_def(ts_tab/_, _, _)).

% Pipe: split on the literal "|" (string split, not RegExp, so no escaping).
test(csv_pipe_in_code, [setup(cleanup_psv), cleanup(cleanup_psv)]) :-
    write_psv_fixture(PF),
    source(csv, ts_pipe, [ csv_file(PF), has_header(true), delimiter('|') ]),
    once(is_dynamic_source(ts_pipe/3)),
    once(typescript_target:compile_predicate_to_typescript(ts_pipe/3, [], Code)),
    has(Code, '.split("|")'),
    catch(delete_file(PF), _, true).

test(csv_pipe_node_execution, [ condition(node_available),
                                setup(cleanup_psv), cleanup(cleanup_psv) ]) :-
    write_psv_fixture(PF),
    source(csv, ts_pipe, [ csv_file(PF), has_header(true), delimiter('|') ]),
    once(typescript_target:compile_predicate_to_typescript(ts_pipe/3, [], Code)),
    ts_write_run(Code, [], OutAll),
    OutAll == 'alice:25:nyc bob:30:sf charlie:35:la',
    catch(delete_file(PF), _, true).

cleanup_psv :- retractall(dynamic_source_compiler:dynamic_source_def(ts_pipe/_, _, _)).

% ============================================================================
% JSON columns() projection (G-P9 polish)
% ============================================================================

% The emitted script carries the projection array in declared order.
test(json_projection_in_code, [setup(cleanup_json), cleanup(cleanup_json)]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_product, [ json_file(JF), columns([price, name, id]), arity(3) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_product/3, [], Code)),
    has(Code, '["price","name","id"]').

% Reordering columns() reorders the emitted rows (before/after: the default
% [id,name,price] order is covered by json_node_execution above).
test(json_projection_reorder, [ condition(node_available),
                                setup(cleanup_json), cleanup(cleanup_json) ]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_product, [ json_file(JF), columns([price, name, id]), arity(3) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_product/3, [], Code)),
    ts_write_run(Code, [], Out),
    Out == '999:Laptop:P001 25:Mouse:P002 75:Keyboard:P003'.

% Subset projection: pick a single key (arity 1) out of the objects.
test(json_projection_subset, [ condition(node_available),
                               setup(cleanup_json), cleanup(cleanup_json) ]) :-
    abs('test_data/test_products.json', JF),
    once(source(json, ts_product, [ json_file(JF), columns([name]), arity(1) ])),
    once(typescript_target:compile_predicate_to_typescript(ts_product/1, [], Code)),
    ts_write_run(Code, [], Out),
    Out == 'Laptop Mouse Keyboard'.

:- end_tests(typescript_source).

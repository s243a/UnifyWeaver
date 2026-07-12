:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% "Iterate tuples, get a struct per item" -- the loop-over-a-collection
% shape, served by `foreach` over a repetition field whose elements have
% MORE THAN ONE field. Each element is a tuple ($1, $2, ...), and the
% body (a full scalar action sequence, since #recbind-loop) can decode
% any field into a structured record via a grammar. This is the
% recommended pattern for "loop and process each item as an object";
% iterating a hash table built with `arr[key]=val` (assoc for-in) with
% real per-entry work is a separate, larger surface (it needs the loop
% key and `arr[k]` value plumbed through the expression model).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar: sq(X) -> pair(X, X*X) -- decode an element field into a struct
sq(X, pair(X, Y)) :- Y is X * X.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_foreach_tuples).

% A tuple element: foreach over rep(i64 i64) exposes $1 and $2 (the two
% fields of the current element) -- the (key, value) pair. 2 elements
% (10,100), (20,200): ta = 30, tb = 300.
test(foreach_tuple_fields, [condition(clang_available)]) :-
    ft_dir(Dir),
    Src = "BEGIN { BINFMT = \"i64 rep4(i64 i64)\" }\n\c
           { foreach { ta += $1 ; tb += $2 } }\n\c
           END { print ta, tb }\n",
    build_run(Dir, 'ftt', Src, [1, 2, 10, 100, 20, 200], "30 300\n"),
    !.

% Struct per tuple: decode the first field of each element into a record
% via a grammar, alongside the raw tuple fields. sq(a) = pair(a, a*a);
% tsq sums a*a. (10,100),(20,200): ta=30, tb=300, tsq=100+400=500.
test(foreach_tuple_to_struct, [condition(clang_available)]) :-
    ft_dir(Dir),
    directory_file_path(Dir, 'sq.wamo', Wamo),
    write_wam_object([user:sq/2], [wamo_entries([sq/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 rep4(i64 i64)\" ; DYNLOAD = \"~w\" }\n\c
         { foreach { (n, sqr) = dyncall@sq($1) as (i64 i64) ; ta += $1 ; tb += $2 ; tsq += sqr } }\n\c
         END { print ta, tb, tsq }\n", [Wamo]),
    build_run(Dir, 'fts', Src, [1, 2, 10, 100, 20, 200], "30 300 500\n"),
    !.

% Struct-view per tuple: the record view reads the decoded struct as the
% current record inside its block ($1 = a, $2 = a*a from sq), while the
% element's own fields are captured first. sum of a*a = 500.
test(foreach_tuple_struct_view, [condition(clang_available)]) :-
    ft_dir(Dir),
    directory_file_path(Dir, 'sq.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:sq/2], [wamo_entries([sq/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 rep4(i64 i64)\" ; DYNLOAD = \"~w\" }\n\c
         { foreach { dyncall@sq($1) as (i64 i64) { tsq += $2 } } }\n\c
         END { print tsq }\n", [Wamo]),
    build_run(Dir, 'ftv', Src, [1, 2, 10, 100, 20, 200], "500\n"),
    !.

:- end_tests(plawk_foreach_tuples).

% --- helpers ---------------------------------------------------------------

ft_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_foreach_tuples', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text), close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_rep_record(Path, Words) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(V, Words), write_i64_le(Out, V)),
        close(Out)).

build_run(Dir, Name, Src, Words, Expected) :-
    write_prog(Dir, Name, Src),
    directory_file_path(Dir, Name, Prog),
    atom_concat(Prog, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog, '_in.bin', Input),
    write_rep_record(Input, Words),
    run_bin(Bin, [Input], Out, 0),
    assertion(Out == Expected).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

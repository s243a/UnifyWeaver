:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Structured-return record binds/views inside a LOOP body -- the second
% step of routing record binds through every block body (if-branches
% were the first). A `foreach { ... }` over a record's repetition
% elements can call a grammar per element and destructure or view the
% returned record: the foreach body already lowers through the scalar
% action-sequence walker (which handles dynrec_bind), so once branch/loop
% body VALIDATION accepts the bind and the record-view desugar recurses
% into the loop body, both surfaces compile.
%
% (`for (k in arr)` -- the assoc for-in -- stays print-only by
% construction: its body is a per-key print plan, not a scalar action
% sequence, so record binds there remain a separate driver concern.)

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% grammar object: rec(X) -> pair(X, X+1)
recp(X, pair(X, Y)) :- Y is X + 1.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_rec_loop).

% The record-view desugar now recurses into foreach bodies: a view there
% becomes a hidden destructure + rewritten block, both inside the loop.
test(view_in_foreach_desugars) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64 rep4(i64)\" ; DYNLOAD = \"l.wamo\" }\n\c
         { foreach { dyncall@recp($1) as (i64 i64) { total += $1 } } }\n\c
         END { print total }\n",
        program(_, Rules0, _)),
    plawk_native_codegen:plawk_resolve_dynrec_view_rules(Rules0, Rules),
    Rules = [rule(_, [foreach(Body)])],
    assertion(memberchk(dynrec_bind(_, dyncall_named(recp, [field(1)]), _),
        Body)),
    assertion(\+ ( member(A, Body), A = dynrec_view(_, _, _) )),
    !.

% Destructure inside foreach, end to end. rep elements 10, 20, 30;
% recp(X) = pair(X, X+1); total += n(=X) -> 60.
test(destructure_in_foreach_runs, [condition(clang_available)]) :-
    rl_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 rep4(i64)\" ; DYNLOAD = \"~w\" }\n\c
         { foreach { (n, m) = dyncall@recp($1) as (i64 i64) ; total += n } }\n\c
         END { print total }\n", [Wamo]),
    build_run_rep(Dir, 'rld', Src, [1, 3, 10, 20, 30], "60\n"),
    !.

% Record VIEW inside foreach: total sums $1 (=X), s sums $2 (=X+1).
% elements 10, 20, 30 -> total 60, s 11+21+31 = 63.
test(view_in_foreach_runs, [condition(clang_available)]) :-
    rl_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 rep4(i64)\" ; DYNLOAD = \"~w\" }\n\c
         { foreach { dyncall@recp($1) as (i64 i64) { total += $1 ; s += $2 } } }\n\c
         END { print total, s }\n", [Wamo]),
    build_run_rep(Dir, 'rlv', Src, [1, 3, 10, 20, 30], "60 63\n"),
    !.

% Combined: a view inside an if inside foreach -- the desugar recurses
% through both, and the per-element guard selects which elements view.
% elements 10, 20, 30; guard $1 > 15 keeps 20, 30; total = 20 + 30 = 50.
test(view_in_if_in_foreach_runs, [condition(clang_available)]) :-
    rl_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64 rep4(i64)\" ; DYNLOAD = \"~w\" }\n\c
         { foreach { if ($1 > 15) { dyncall@recp($1) as (i64 i64) { total += $1 } } } }\n\c
         END { print total }\n", [Wamo]),
    build_run_rep(Dir, 'rliv', Src, [1, 3, 10, 20, 30], "50\n"),
    !.

:- end_tests(plawk_dyncall_rec_loop).

% --- helpers ---------------------------------------------------------------

rl_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rec_loop', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text), close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

% one record: a flat i64 sequence [id, count, elem1, ...] for
% BINFMT "i64 rep4(i64)".
write_rep_record(Path, Words) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(V, Words), write_i64_le(Out, V)),
        close(Out)).

build_run_rep(Dir, Name, Src, Words, Expected) :-
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

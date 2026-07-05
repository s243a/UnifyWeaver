:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) roadmap item 4, record-view target: the awk-like surface
% for structured returns. `dyncall[@name](args) as (T1 ... Tn) { Body }`
% makes the returned compound the current record inside Body, so `$1`,`$2`
% read its fields. It desugars to a destructure into hidden temporaries
% plus Body with `$k` rewritten to the k-th temporary -- no field-pointer
% repoint, riding the destructure machinery landed earlier.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

% rec(X) returns a 2-field compound pair(X, X+0.5): field 0 i64, field 1 f64.
rec(X, R) :- H is X + 0.5, R = pair(X, H).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dyncall_record_view).

% The view block parses to its own node with the body carrying $1/$2.
test(view_parse) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         { dyncall@rec($1) as (i64 f64) { total += $1 ; sum += $2 } }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(dynrec_view(dyncall_named(rec, [field(1)]), [i64, f64],
        [add(var(total), field(1)), add(var(sum), field(2))]), Actions),
    !.

% The desugar rewrites $k in the body to per-site temporaries and prepends
% the destructure bind; no field references survive.
test(view_desugars_to_bind) :-
    Rules = [rule(always, [dynrec_view(dyncall_named(rec, [field(1)]),
        [i64, f64], [add(var(total), field(1)), add(var(sum), field(2))])])],
    plawk_native_codegen:plawk_resolve_dynrec_view_rules(Rules, [rule(always, Out)]),
    Out = [dynrec_bind(Temps, dyncall_named(rec, [field(1)]), [i64, f64]),
           add(var(total), var(T1)), add(var(sum), var(T2))],
    Temps = [T1, T2],
    % the body no longer references $1/$2 (only the bind's call arg does)
    \+ plawk_native_codegen:plawk_term_has_field_ref(
        [add(var(total), var(T1)), add(var(sum), var(T2))]),
    !.

% A view whose body reads a field outside 1..nfields does not desugar
% (the record has no such field) -> the pass leaves it uncompilable.
test(view_out_of_range_field_rejected) :-
    Rules = [rule(always, [dynrec_view(dyncall(_), [i64],
        [add(var(t), field(3))])])],
    \+ plawk_native_codegen:plawk_resolve_dynrec_view_rules(Rules, _),
    !.

% Full round trip, i64 field via $1: sum field 0 over inputs 10, 20 -> 30.
test(view_i64_field_runs, [condition(clang_available)]) :-
    rv_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:rec/2], [wamo_entries([rec/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { dyncall@rec($1) as (i64 f64) { total += $1 } }\n\c
         END { print total }\n", [Wamo]),
    build_run(Dir, 'rvi', Src, [10, 20], Out),
    assertion(Out == "30\n"),
    !.

% Full round trip, f64 field via $2: sum field 1 (X+0.5) over 10, 20 ->
% 10.5 + 20.5 = 31.
test(view_f64_field_runs, [condition(clang_available)]) :-
    rv_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:rec/2], [wamo_entries([rec/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { dyncall@rec($1) as (i64 f64) { sum += $2 } }\n\c
         END { print sum }\n", [Wamo]),
    build_run(Dir, 'rvf', Src, [10, 20], Out),
    assertion(Out == "31\n"),
    !.

:- end_tests(plawk_dyncall_record_view).

% --- helpers ---------------------------------------------------------------

rv_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dyncall_rview', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Values, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    directory_file_path(Dir, 'in.bin', Input),
    write_i64_records(Input, Values),
    run_bin(Bin, [Input], Out, 0).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_i64_records(Path, Values) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(V, Values), write_i64_le(Out, V)),
        close(Out)).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(null), process(Pid)]),
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

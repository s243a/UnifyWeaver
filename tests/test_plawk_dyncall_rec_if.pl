:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Structured-return record binds/views inside an if-branch. The scalar
% action-sequence walker already lowers `dynrec_bind` wherever it appears
% (its slot + object-call IR is branch-position-independent); the only
% gap was branch-body VALIDATION, which did not list dynrec_bind among
% the actions a branch may hold. With that closed, a destructure or a
% record view can sit inside `if { ... }` / `else { ... }` -- the first
% step of routing record binds through every block body (loop bodies
% are the next).

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

:- begin_tests(plawk_dyncall_rec_if).

% A destructure inside an if-branch validates as a scalar rule body.
test(destructure_in_if_is_compilable_surface) :-
    plawk_parse_string(
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"l.wamo\" }\n\c
         { if ($1 > 0) { (n, m) = dyncall@recp($1) as (i64 i64) ; total += n } }\n\c
         END { print total }\n",
        program(_, Rules, _)),
    Rules = [rule(_, [if(_, Then, _)])],
    assertion(memberchk(dynrec_bind([n, m], _, [i64, i64]), Then)),
    % the branch-body validator now accepts the bind
    assertion(forall(member(A, Then),
        plawk_native_codegen:plawk_scalar_rule_body_plain_action(A))),
    !.

% Destructure in an if-branch, end to end. Records 3, -1, 5: the guard
% keeps 3 and 5; recp(X) = pair(X, X+1); total += n(=X) -> 3 + 5 = 8.
test(destructure_in_if_runs, [condition(clang_available)]) :-
    ri_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { if ($1 > 0) { (n, m) = dyncall@recp($1) as (i64 i64) ; total += n } }\n\c
         END { print total }\n", [Wamo]),
    build_run_i64(Dir, 'rif', Src, [3, -1, 5], "8\n"),
    !.

% Record VIEW inside an if-branch (desugars to a hidden destructure +
% $k rewrite): total sums $1 (=X) over kept records, s sums $2 (=X+1).
% 3, -1, 5 -> total = 8, s = 4 + 6 = 10.
test(view_in_if_runs, [condition(clang_available)]) :-
    ri_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { if ($1 > 0) { dyncall@recp($1) as (i64 i64) { total += $1 ; s += $2 } } }\n\c
         END { print total, s }\n", [Wamo]),
    build_run_i64(Dir, 'rifv', Src, [3, -1, 5], "8 10\n"),
    !.

% The else branch also takes a bind: guard picks the branch, each side
% binds and accumulates. 3 (>0 -> a), -1 (else -> b): a += 3, b += (-1)+1=0
% ... use recp so else adds X+1 of the negative. 3 -> a=3; -1 -> b += 0.
test(bind_in_else_runs, [condition(clang_available)]) :-
    ri_dir(Dir),
    directory_file_path(Dir, 'recp.wamo', Wamo),
    ( exists_file(Wamo) -> true
    ; write_wam_object([user:recp/2], [wamo_entries([recp/2])], Wamo) ),
    format(string(Src),
        "BEGIN { BINFMT = \"i64\" ; DYNLOAD = \"~w\" }\n\c
         { if ($1 > 0) { (n, m) = dyncall@recp($1) as (i64 i64) ; a += n } \c
           else { (p, q) = dyncall@recp($1) as (i64 i64) ; b += q } }\n\c
         END { print a, b }\n", [Wamo]),
    % 3 -> a += 3 ; -1 -> q = (-1)+1 = 0, b += 0 ; 5 -> a += 5. a=8, b=0.
    build_run_i64(Dir, 'rife', Src, [3, -1, 5], "8 0\n"),
    !.

:- end_tests(plawk_dyncall_rec_if).

% --- helpers ---------------------------------------------------------------

ri_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_rec_if', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Text) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, Out, [encoding(utf8)]),
        write(Out, Text), close(Out)).

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

write_i64_records(Path, Values) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(V, Values), write_i64_le(Out, V)),
        close(Out)).

build_run_i64(Dir, Name, Src, Values, Expected) :-
    write_prog(Dir, Name, Src),
    directory_file_path(Dir, Name, Prog),
    atom_concat(Prog, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog, '_in.bin', Input),
    write_i64_records(Input, Values),
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

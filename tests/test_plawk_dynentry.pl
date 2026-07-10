:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Surface B (DYNENTRY): declaration-bound library names. For a fixed
% DYNLOAD shipping a known entry family, bind names once and call them
% like ordinary functions -- `parse($1)` instead of `dyncall@parse($1)`.
% A declared name is RESERVED for the compiled object (parse-time
% rewrite to the named-entry machinery, so the PC still resolves once
% at startup); declaring a name userspace also defines is a parse
% error, never silent shadowing.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

square(X, R) :- atom_number(X, N), R is N * N.
cube(X, R)   :- atom_number(X, N), R is N * N * N.
halve(X, R)  :- atom_number(X, N), R is N / 2.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_dynentry).

% DYNENTRY declarations rewrite bare calls to the named-entry nodes at
% parse time -- i64 and float positions alike.
test(dynentry_rewrites_bare_calls) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         DYNENTRY square, halve\n\c
         { t += square($1) ; f += float(halve($1)) }\nEND { print t }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(add(var(t), dyncall_named(square, [field(1)])), Actions),
    memberchk(add(var(f), float_dyncall_named(halve, [field(1)])), Actions),
    !.

% An undeclared bare call stays an ordinary compiled-foreign call node.
test(undeclared_calls_untouched) :-
    plawk_parse_string(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         DYNENTRY square\n\c
         { t += square($1) + other($1) }\nEND { print t }\n",
        program(_, [rule(_, Actions)], _)),
    memberchk(add(var(t), add_i64(dyncall_named(square, [field(1)]),
        prolog_call(other, [field(1)]))), Actions),
    !.

% Reservation: a DYNENTRY name an @prolog block also defines throws --
% never silent shadowing.
test(dynentry_over_block_predicate_throws,
     [throws(error(plawk_dynentry_reserved(score), _))]) :-
    plawk_parse_source(
        "@prolog\nscore(X, R) :- R is X + 1.\n@end\n\c
         BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         DYNENTRY score\n\c
         { t += score($1) }\nEND { print t }\n", _, _).

% Reservation: surface builtin/keyword names are off limits.
test(dynentry_over_builtin_throws,
     [throws(error(plawk_dynentry_reserved(length), _))]) :-
    plawk_parse_source(
        "BEGIN { DYNLOAD = \"lib.wamo\" }\n\c
         DYNENTRY length\n\c
         { t += length($0) }\nEND { print t }\n", _, _).

% Full round trip: two declared entries called like ordinary functions,
% resolving against the DYNLOAD object's entry table (squares 4+9 +
% cubes 8+27 = 48 over 2,3).
test(dynentry_calls_resolve, [condition(clang_available)]) :-
    de_dir(Dir),
    directory_file_path(Dir, 'lib.wamo', Wamo),
    write_wam_object([user:square/2, user:cube/2],
        [wamo_entries([square/2, cube/2])], Wamo),
    format(string(Src),
        "BEGIN { DYNLOAD = \"~w\" }\n\c
         DYNENTRY square, cube\n\c
         { total += square($1) + cube($1) }\n\c
         END { print total }\n", [Wamo]),
    build_run(Dir, 'de', Src, "2\n3\n", Out),
    assertion(Out == "48\n"),
    !.

:- end_tests(plawk_dynentry).

% --- helpers ---------------------------------------------------------------

de_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dynentry', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
    run_bin(Bin, [Input], Out, 0).

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
